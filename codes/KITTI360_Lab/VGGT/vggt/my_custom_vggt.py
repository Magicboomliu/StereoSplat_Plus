import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
# from vggt.heads.camera_head import CameraHead
from vggt.heads.camera_head_extrin import CameraHeadExtrin

from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri,extri_to_pose_encoding
from vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.losses.losses import depth_loss,camera_loss,pcd_loss
from vggt.metrics.metrics import compute_rra_rta_absolute,compute_depth_mae_mse,compute_pcd_mae_mse,convert_depth_to_disp
import os
import numpy as np
import skimage.io

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head_extrin = CameraHeadExtrin(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        # self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def prepare_input_data(self,batch):

        '''
        Prepared all following data for training:
        
            (1) First Normalize all the camera pose into the first frame Camera Coordinate.
            (2) Convert the Camera Pose into Quant.
            (3) Get the Point Cloud for each input.
            (4) Get the depth estimation.

        '''
        def compute_relative_camera_pose(cam2world: torch.Tensor):
            """
            输入: cam2world [B, V, 4, 4]
            输出: relative_cam2world [B, V, 4, 4], 相对于每个 batch 的第一个视角
            """
            B, V, _, _ = cam2world.shape
            relative_cam2world = torch.empty_like(cam2world)

            for batch_idx in range(B):
                cam2world_batch = cam2world[batch_idx]  # (V, 4, 4)
                ref_cam2world = cam2world_batch[0]      # (4, 4)
                ref_world2cam = torch.inverse(ref_cam2world)  # (4, 4)

                # 计算该 batch 内所有 view 相对于第一个 view 的 cam2world
                relative_cam2world[batch_idx] = torch.matmul(ref_world2cam, cam2world_batch)

            return relative_cam2world
        
        def construct_intrinsics_matrix(input_fx, input_fy, input_cx, input_cy):
            """
            输入:
                input_fx, input_fy, input_cx, input_cy: (1, 4)
            输出:
                intrinsics: (1, 4, 3, 3)
            """
            B, V = input_fx.shape  # 一般 B = 1, V = 4

            device = input_fx.device
            dtype = input_fx.dtype

            intrinsics = torch.zeros((B, V, 3, 3), device=device, dtype=dtype)

            intrinsics[:, :, 0, 0] = input_fx
            intrinsics[:, :, 1, 1] = input_fy
            intrinsics[:, :, 0, 2] = input_cx
            intrinsics[:, :, 1, 2] = input_cy
            intrinsics[:, :, 2, 2] = 1.0

            return intrinsics

        def depth_to_world_points(depth, K, cam2world):
            """
            将深度图转换为世界坐标下的点云图。

            输入:
                depth:      (B, V, H, W) 深度图，单位为米
                K:          (B, V, 3, 3) 相机内参
                cam2world:  (B, V, 4, 4) cam2world pose

            输出:
                world_points: (B, V, 3, H, W) 每个像素点在世界坐标下的3D位置
            """
            B, V, H, W = depth.shape
            device = depth.device

            # === 1. 构造像素中心坐标网格 ===
            y, x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device) + 0.5,
                torch.arange(W, dtype=torch.float32, device=device) + 0.5,
                indexing='ij'
            )  # x, y shape: (H, W)

            ones = torch.ones_like(x)
            pixel_coords = torch.stack((x, y, ones), dim=-1)  # (H, W, 3)
            pixel_coords = pixel_coords.view(-1, 3).T  # (3, H*W)

            # === 2. 扩展 pixel_coords 到 (B, V, 3, H*W) ===
            pixel_coords = pixel_coords[None, None, :, :].expand(B, V, -1, -1)  # (B, V, 3, H*W)

            # === 3. 计算内参逆矩阵 ===
            K_inv = torch.inverse(K)  # (B, V, 3, 3)

            # === 4. 投影到相机坐标系 ===
            rays = torch.matmul(K_inv, pixel_coords)  # (B, V, 3, H*W)

            # === 5. 乘以深度得到相机坐标系下的点 ===
            depth_flat = depth.view(B, V, -1)  # (B, V, H*W)
            cam_points = rays * depth_flat.unsqueeze(2)  # (B, V, 3, H*W)

            # === 6. 添加齐次维度 (4D) ===
            ones = torch.ones_like(depth_flat)
            cam_points_homo = torch.cat([cam_points, ones.unsqueeze(2)], dim=2)  # (B, V, 4, H*W)

            # === 7. 应用 cam2world 得到世界坐标 ===
            world_points_homo = torch.matmul(cam2world, cam_points_homo)  # (B, V, 4, H*W)

            # === 8. 去掉齐次分量，并 reshape 成 (B, V, 3, H, W) ===
            world_points = world_points_homo[:, :, :3, :].view(B, V, 3, H, W)

            return world_points
        
        input_dict = dict()
        input_dict['input_rgb'] =torch.concat(batch['inputs']['rgb'],dim=1)  #(1,4,3,112,518)
        
        input_c2ws = torch.concat(batch['inputs_pix']['c2w'],dim=1) #torch.Size([1, 4, 4, 4])
        input_fx = torch.cat(batch['inputs_pix']['fx'],dim=1) #(1,4)
        input_fy = torch.cat(batch['inputs_pix']['fy'],dim=1) #(1,4)
        input_cx = torch.cat(batch['inputs_pix']['cx'],dim=1) #(1,4)
        input_cy = torch.cat(batch['inputs_pix']['cy'],dim=1) #(1,4)
        input_depth = torch.cat(batch['inputs_pix']['depth_m'],dim=1) #[1,4,H,W]
        input_depth_sparse = torch.cat(batch['inputs_pix']['sparse_gt_depth'],dim=1) ##[1,4,H,W]
        
        valid_depth_mask = input_depth_sparse> 0
        valid_depth_mask =  valid_depth_mask.float()
        #FIXME Here
        #input_fusion_depth = valid_depth_mask * input_depth_sparse + (1-valid_depth_mask) * input_depth
        input_fusion_depth = input_depth
        
        
        input_c2ws_relative_matrix = compute_relative_camera_pose(input_c2ws) #(B,V,4,4)
        
        input_intrinsic_matrix = construct_intrinsics_matrix(input_fx=input_fx,
                                    input_fy=input_fy,
                                    input_cx=input_cx,
                                    input_cy=input_cy)

        input_pos_enc = extri_to_pose_encoding(extrinsics=input_c2ws_relative_matrix) #(1,4,7)

        input_pointmap = depth_to_world_points(depth=input_fusion_depth,
                                                K=input_intrinsic_matrix,
                                                cam2world=input_c2ws_relative_matrix)

        
        input_dict['input_c2ws_rel_mat'] = input_c2ws_relative_matrix
        input_dict['input_intrinsic_mat'] = input_intrinsic_matrix
        input_dict['input_pos_enc'] = input_pos_enc
        input_dict['input_pointmap'] = input_pointmap
        input_dict['input_depth'] = input_fusion_depth
        
        return input_dict
        
    
    def forward(self, batch,mode='train',cfg=None):
        '''Prepared the Following Input data'''
        
        input_dict = self.prepare_input_data(batch)
        images = input_dict['input_rgb']
        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

                
        
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head_extrin is not None:
                pose_enc_list = self.camera_head_extrin(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf


        
        if mode =='train' or mode=='val':

            bs,vs,rgb_channel,cur_h,cur_w = images.shape[:5]
            
            predictions["images"] = images
            
            predicted_pose_enc = predictions["pose_enc"]

            predicted_depth = predictions['depth'].squeeze(-1).reshape(bs*vs,1,cur_h,cur_w)
            predicted_depth_conf = predictions["depth_conf"].reshape(bs*vs,1,cur_h,cur_w)
            
            predicted_pcd = predictions['world_points'].permute(0,1,4,2,3).reshape(bs*vs,3,cur_h,cur_w)
            predicted_pcd_conf = predictions['world_points_conf'].reshape(bs*vs,1,cur_h,cur_w)
            
            '''
                input_dict['input_c2ws_rel_mat'] = input_c2ws_relative_matrix
                input_dict['input_intrinsic_mat'] = input_intrinsic_matrix
                input_dict['input_pos_enc'] = input_pos_enc
                input_dict['input_pointmap'] = input_pointmap
                input_dict['input_depth']
            
            '''
            
            gt_input_depth = input_dict['input_depth'].reshape(bs*vs,1,cur_h,cur_w)
            gt_pcd = input_dict['input_pointmap'].reshape(bs*vs,3,cur_h,cur_w)
            gt_pos_enc = input_dict['input_pos_enc']

            
            # Get the Loss Here
            # ======================== losses ======================== #
            loss = 0.0
            loss_terms = {}
            def set_loss(key, split, loss_value, loss_weight=1.0):
                loss_terms[f"{split}/loss_{key}"] = loss_value.item()
                loss_terms[f"{split}/loss_{key}_w"] = loss_value.item() * loss_weight  
            
            if cfg.loss_args.use_depth_loss and cfg.loss_args.depth_conf_loss:
                depth_loss_with_conf = depth_loss(pred_depth=predicted_depth,
                                                  gt_depth=gt_input_depth,
                                                  sigma_d=predicted_depth_conf,
                                                  alpha=cfg.loss_args.alpha_weight)
                
                set_loss(key="depth_loss_with_conf",
                         split=mode,
                         loss_value=depth_loss_with_conf,
                         loss_weight=cfg.loss_args.depth_loss_weight)
                
                loss +=depth_loss_with_conf* cfg.loss_args.depth_loss_weight
            
            elif cfg.loss_args.use_depth_loss and not cfg.loss_args.depth_conf_loss:
                pass
            
            
            if cfg.loss_args.use_pcd_loss and cfg.loss_args.pcd_conf_loss:
                pcd_loss_with_conf = pcd_loss(pred_pcd=predicted_pcd,
                         gt_pcd=gt_pcd,
                         sigma_p=predicted_pcd_conf,
                         alpha=cfg.loss_args.alpha_weight)
                
                loss +=pcd_loss_with_conf* cfg.loss_args.pcd_loss_weight
                set_loss(key="pcd_loss_with_conf",
                         split=mode,
                         loss_value=pcd_loss_with_conf,
                         loss_weight=cfg.loss_args.pcd_conf_loss)


            elif cfg.loss_args.use_pcd_loss and not cfg.loss_args.pcd_conf_loss:
                pass
            
            if cfg.loss_args.use_pos_enc_loss:
                camera_loss_data = camera_loss(pred_camera=predicted_pose_enc, 
                            gt_camera=gt_pos_enc)
                loss +=camera_loss_data* cfg.loss_args.pos_enc_loss_weight

                set_loss(key="camera_loss_data",
                         split=mode,
                         loss_value=camera_loss_data,
                         loss_weight=cfg.loss_args.pos_enc_loss_weight)

        if mode=='train':
            return predictions,loss,loss_terms
        elif mode =='val':
            return predictions,loss,loss_terms,input_dict

    
    def validation_step(self,batch,val_results_saved_dir,cfg=None):
        
        with torch.no_grad():
            predictions,loss,loss_terms,input_dict= self.forward(batch,mode='val',cfg=cfg) 
                
        batch_data_for_eval= {
            "pred_pos_enc": predictions["pose_enc"],
            "predicted_depth": predictions['depth'].squeeze(-1),
            "predicted_depth_conf":predictions["depth_conf"],
            "predicted_pcd" : predictions['world_points'].permute(0,1,4,2,3),
            "predicted_pcd_conf":predictions['world_points_conf'],
            
            "gt_input_depth":input_dict['input_depth'],
            "gt_pcd":input_dict['input_pointmap'],
            "gt_pos_enc":input_dict['input_pos_enc'],
            "input_images":input_dict['input_rgb']   
        }
        
        output_rgb_meter_dict = self.save_val_results(batch_data_for_eval=batch_data_for_eval,
                              saved_dir=val_results_saved_dir,
                              cfg=cfg)
        
        return predictions,loss,loss_terms,input_dict,output_rgb_meter_dict
    
    
    def save_val_results(self,batch_data_for_eval,saved_dir,cfg):
        output_rgb_meter_dict = dict()
        
        input_images = batch_data_for_eval['input_images']
        
        predicted_pos_enc = batch_data_for_eval["pred_pos_enc"]
        gt_pos_enc = batch_data_for_eval['gt_pos_enc']
        
        pred_cam2world = pose_encoding_to_extri(predicted_pos_enc)
        gt_cam2world = pose_encoding_to_extri(gt_pos_enc)

        rra, rta = compute_rra_rta_absolute(est=pred_cam2world,
                                 gt=gt_cam2world)
        
        predicted_depth = batch_data_for_eval["predicted_depth"]
        predicted_depth_conf = batch_data_for_eval["predicted_depth_conf"]
        gt_depth = batch_data_for_eval["gt_input_depth"]
        
        
        depth_mae, depth_mse = compute_depth_mae_mse(depth_pred=predicted_depth,
                              depth_gt=gt_depth)
        
        pred_pcd =  batch_data_for_eval["predicted_pcd"]
        pred_pcd_conf = batch_data_for_eval['predicted_pcd_conf']
        gt_pcd = batch_data_for_eval["gt_pcd"]
        
        pcd_mae, pcd_mse = compute_pcd_mae_mse(est_pcd=pred_pcd,
                            gt_pcd=gt_pcd)
        
        
        output_rgb_meter_dict['rra'] = rra.data.data.item()
        output_rgb_meter_dict['rta'] = rta.data.item()
        output_rgb_meter_dict['depth_mae'] = depth_mae.data.item()
        output_rgb_meter_dict['depth_mse'] = depth_mse.data.item()
        
        output_rgb_meter_dict['pcd_mae'] = pcd_mae.data.item()
        output_rgb_meter_dict['pcd_mse'] = pcd_mse.data.item()

        # saved into images.
        os.makedirs(saved_dir,exist_ok=True)

        if cfg.validation_vis_progress:
            
            bs,view_num,cur_height,cur_width = predicted_depth.shape
            
            for idx in range(view_num):
                predict_depth_per = predicted_depth[0][idx].cpu().numpy()
                predict_depth_per_vis = convert_depth_to_disp(depth=predict_depth_per)
                skimage.io.imsave(os.path.join(saved_dir,"pred_depth_{}.png".format(idx)),
                                  predict_depth_per_vis)
                
                
                gt_depth_per = gt_depth[0][idx].cpu().numpy()
                gt_depth_per_vis = convert_depth_to_disp(depth=gt_depth_per)
                skimage.io.imsave(os.path.join(saved_dir,"gt_depth_{}.png".format(idx)),
                                  gt_depth_per_vis)
                
                
                skimage.io.imsave(os.path.join(saved_dir,'input_image_{}.png'.format(idx)),
                                   (input_images.squeeze(0)[idx].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            
        
        
        return output_rgb_meter_dict
        

if __name__=="__main__":
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    import matplotlib.pyplot as plt
    import pickle
    
    def saved_into_pickle(data_dict,path):
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)
            
    
    
    input_images_path = ["/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0002_sync/image_00/data_rect/0000004391.png",
                         "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0002_sync/image_01/data_rect/0000004391.png",
                         "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0002_sync/image_00/data_rect/0000004392.png",
                         "/data1/StereoDatasets/KITTI/KITTI360/data_2d_raw/2013_05_28_drive_0002_sync/image_01/data_rect/0000004392.png"
                         ]
    
    
    input_images = load_and_preprocess_images(input_images_path).to("cuda")
    
    
    
    vggt = VGGT().cuda()
    model_path = "/data1/zliu/foundation_model/model.pt"
    ckpt = torch.load(model_path)
    vggt.load_state_dict(ckpt,strict=False)


    predictions = vggt(input_images)
    pose_enc = predictions['pose_enc']
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, input_images.shape[-2:])
    depths = predictions['depth']
    pcd = predictions['world_points']
    depth_conf = predictions['depth_conf']
    images = predictions['images']
    
    
    saved_dict = dict()
    saved_dict['extrinsic'] = extrinsic.detach().cpu().numpy()
    saved_dict['intrinsic'] = intrinsic.detach().cpu().numpy()
    saved_dict['depth'] = depths.detach().cpu().numpy()
    saved_dict['world_points'] = pcd.detach().cpu().numpy()
    saved_dict['depth_conf'] = depth_conf.detach().cpu().numpy()
    saved_dict['images'] = images.detach().cpu().numpy()
    
    saved_into_pickle(data_dict=saved_dict,
                      path="/home/zliu/Project2025/example_output.pkl")
    
    




    pass
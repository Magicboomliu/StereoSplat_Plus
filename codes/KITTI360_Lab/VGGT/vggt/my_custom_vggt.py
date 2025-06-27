import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.load_fn import load_and_preprocess_images


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        # self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
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

        # if self.track_head is not None and query_points is not None:
        #     track_list, vis, conf = self.track_head(
        #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
        #     )
        #     predictions["track"] = track_list[-1]  # track of the last iteration
        #     predictions["vis"] = vis
        #     predictions["conf"] = conf

        predictions["images"] = images

        return predictions




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
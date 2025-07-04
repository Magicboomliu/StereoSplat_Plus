## Report of the reliability of existing monocular depth estimation on KITTI-360 Dataset


#### Candidates Monocular Depth Estimator  
- DepthAnythingV2 (MonoDepth)
- Metric3D V2 (MonoDepth)
- NMRFStereo (Stereo Matching)

#### Metrics for KITTI360 sequence 0000:  

Note: all the predicted depth are using the alignment algorthims for scale matching, we report the average `MAE` and average `MSE` for performance evaluation. 
---
|    TYPE   	|  Network Method 	|  Alignmnet Method  	| Averge MAE(Left) 	| Average MSE(Right) 	| Average MAE(Left) 	| Average MSE(Right) 	|
|:---------:	|:---------------:	|:------------------:	|:----------------:	|:------------------:	|:-----------------:	|:------------------:	|
| Monocular 	| DepthAnythingV2 	|    Least Square    	|       2.07       	|        17.85       	|        2.08       	|        18.21       	|
|           	| DepthAnythingV2 	| Medium Scale Align 	|       1.90       	|        22.54       	|        1.93       	|        22.92       	|
|           	|   Metric3D-V2   	|        None        	|       1.56       	|        16.83       	|        1.58       	|        17.12       	|
|           	|   Metric3D-V2   	|    Least Square    	|       1.70       	|        14.96       	|        1.75       	|        15.39       	|
|           	|   Metric3D-V2   	| Medium Scale Align 	|       1.68       	|        16.09       	|        1.73       	|        16.73       	|
|   Stereo  	|   NMRF-Stereo   	|        None        	|       1.19       	|        14.52       	|        1.21       	|        15.14       	|
|   Stereo  	|   NMRF-Stereo   	|        Least Square        	|       1.54       	|        18.34      	|        1.56       	|        18.14       	|
|   Stereo  	|   NMRF-Stereo   	|        Medium Scale Align        	|       1.59       	|        18.84      	|        1.59       	|        18.54       	|

---

```
python rel_depth_to_metirc.py
``` 

The key function is that: 
```
def depth2metric(rel_depth,gt_sparse_depth,valid_mask,align_type='LS'):
    '''
    rel_depth: realtive depeth
    gt_sparse_depth: projected sparse depth
    valid_mask: valid regions contains project lidar
    align_type: alignment manner, selected from the "LS" (Linear Square), "Med"(Median Matching)
    ------------------------------------------------------
    reference: 
    (1) "LS" implementation from Marigold(CVPR2024): https://github.com/prs-eth/Marigold/blob/main/src/util/alignment.py#L8
    (2) "Med" implementation from hierarchical-3d-gaussians(SIGGRAPH2024): https://github.com/graphdeco-inria/hierarchical-3d-gaussians/blob/main/preprocess/make_depth_scale.py#L19
    '''
    
    
    if align_type =="LS":
        aligned_pred, scale, shift = align_depth_least_square(gt_arr=gt_sparse_depth,
                                pred_arr=rel_depth,
                                valid_mask_arr = valid_mask.astype(np.bool_)
                                )
    
    elif align_type =="Med":
        aligned_pred, scale, shift = Med_Scaling_Depth(rel_depth,gt_sparse_depth,valid_mask.astype(np.bool_))
    
    elif align_type =="None":
        aligned_pred = rel_depth
        scale =1
        shift =1
    
    else:
        raise NotImplementedError
    
    return aligned_pred, scale, shift 
```

Usage: 
```
from tools_funcs.metric_depth_alignment.rel_depth_to_metric import depth2metric
```
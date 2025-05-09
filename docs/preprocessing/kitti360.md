# KITTI-360 Dataset Preparation and Preprocssing

### Step1: Dataset Download 
1. Download the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/download.php) dataset.

    Only the following data are required.

    - Left perspective images (124 GB)
    - Left instance masks (2.2 GB)
    - 3D bounding boxes (420 MB)
    - Camera parameters (28 KB)
    - Camera poses (28 MB)

    Make sure the directory structure is the same as below:

    ```bash
    KITTI-360
    ├── calibration         # camera parameters
    ├── data_2d_raw         # perspective images
    ├── data_2d_semantics   # instance masks
    ├── data_3d_bboxes      # 3D bounding boxes
    └── data_poses          # camera poses
    └── data_3d_raw         # LiDAR from Velodye
    ```
2. Make a JSON annotation file for each frame.

    ```bash
    python preprocessing/make_annotations_json/make_annotations.py \
        --root_dirname ROOT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

    A directory named `annotations` will be created as follows.

    ```bash
    KITTI-360
    ├── annotations         # per-frame annotations
    ├── calibration         # camera parameters
    ├── data_2d_raw         # perspective images
    ├── data_2d_semantics   # instance masks
    ├── data_3d_bboxes      # 3D bounding boxes
    └── data_poses          # camera poses
    ```

    Note that the following frames are excluded.

    - Frames without camera poses
    - Frames without instance masks

### Step2: Get the Avaiable filename list 
```
# default save at filenames/kitti360/avaliable_lists

cd  scripts/preprocessing/bins_split/kitti_360/
sh get_avaliable_images_by_sequence.sh
```  


### Step3: Create Bins for feedforward 3DGS  

Note all default using the `center frame`'s LiDAR coordiante as the world coordiante of the entire bin scenes. Each sensors including `CAM_LEFT`, `CAM_RIGHT` and `LIDAR_TOP` 's camera instrinsic and camera pose.  

Each bin info including: 
```
info = {    
    "token": string
    "scene_token": string
    "timestep": fname
    "bin_length": default set is 8.0(maybe longer)
    "sensor_info" : {'LIDAR_TOP': list, 'CAM_LEFT': list, 'CAM_RIGHT': list}
}
``` 
For each sensor, it is a list of all frames inside this bins, start with [center, first, last,...(Others by order)], for each element it is a dict: 
```
{
"data_path": string
"type": sensor_type,
"sample_token":Path(os.path.basename(data_path)).stem,
"sensor2world_translation":sensor_to_world_translation,
"sensor2world_rotation": sensor_to_world_rotation,
"sensor2world_transform": sensor_to_world_transform,
"sensor2lidar_translation":sensor_to_reference_lidar_translation,
"sensor2lidar_rotation":sensor_to_refernece_lidar_rotation,
"sensor2lidar_transform":sensor_to_reference_lidar_transform
}       
```

Related Codes for generating the bins for given sequence filename :  
```
cd  scripts/preprocessing/bins_split/kitti_360/
sh create_bins.sh
```

### Step4: Get the Relative Depth and the Metric Depth for the given dataset.  

#### (1) Depth Estimation
Currently, we implement the `DepthanythingV2` and `Metric3DV2` for relative depth and metric depth respectivly. For model zoo details, please refer to [model zoo](preprocessing/monocular_depth_estimation/Models_Zoo.md). 

- DepthAnything Depth Estimation 
```
cd scripts/preprocessing/depth_estimation/

sh monocular_depth_estimation.sh
```
- Metric3D Depth Estimation

```
cd scripts/preprocessing/depth_estimation/
sh monocular_depth_estimation.sh
```

#### (2) Get projected sparse lidar map 

```
cd scripts/preprocessing/depth_estimation/ 
sh sparse_gt_lidar_projection.sh
```

#### (3) Metric Depth Scale Alignment  

```
python tools_funcs/metric_depth_alignment/rel_depth_to_metric.py
```
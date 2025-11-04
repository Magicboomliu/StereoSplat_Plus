# VolumeFusion Ablation Studies

In this experiemnts, we conduct the following ablation studies.


- 1. Without using the Diffix 3D for image quality enhancement.
    - Exp 1: cost volume branch only version.
    - Exp 2: triplane branch only version.
    - Exp 3: volumefusion version.
    - Exp 4: volumefusion with randomly sampling and randomly stereo pairs.

- 2. With using the Diffix3D for image quality enhancement.
    - Exp 5: using progressive 2 times, 2 views ----> 6 views.
    - Exp 6: using progressive 3 times, 2 views ---> 4 views ---> 6 views.


### Pretrained Models (Weights Google Drive) 

- [Exp 3](https://drive.google.com/drive/folders/1MX-MfpihcIPHwFj6T15HTwMK9kfek2hh?usp=sharing)
- [Exp 4 ](https://drive.google.com/drive/folders/1o0_dbnNs01ytyxjguAe31GZ5KNe83Pqu?usp=sharing)
- [Exp 5,6 Diffusion](https://drive.google.com/file/d/1VnynCkjq_SqQD2z6I-LvtDh-8CSbIiLB/view?usp=sharing)

### Training & Inference & Visualizations

- **Exp 3**: volumefusion trained with first stereo images only.
```
#(1) training the models
cd scripts/train/ablations
sh train_volumefusion_first_2_view.sh

# (2) inference and visualizations
cd scripts/evaluations/ablations
sh volumefusion_train_first_2_views.sh
```


- **Exp 4**:  volumefusion with randomly sampling and randomly stereo 

```
#(1) training the models
cd scripts/train/ablations
sh train_volumefusion_randomly.sh

# (2) inference and visualizations
cd scripts/evaluations/ablations
sh volumefusion_train_random_2_views.sh
```

- **Exp 5**:  using progressive 2 times, 2 views ----> 6 views, note to select `Progressive_Twice_Diffix3D_Once`

```
cd scripts/evaluations/ablations
sh volumefusion_progressive_with_diffix3d.sh 
```

- **Exp 6**:  using progressive 3 times, 2 views ---> 4 views ---> 6 views. note to select `Progressive_Three_Diffix3D_Twice`

```
cd scripts/evaluations/ablations
sh volumefusion_progressive_with_diffix3d.sh 
```
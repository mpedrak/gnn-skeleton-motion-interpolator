# GNN Skeleton Motion Interpolator

3D skeletal motion in-betweening aims to automatically generate realistic transition frames between keyframes, reducing manual work in creating character animation. This repository provides a deep learning approach for this task using Attention-based Graph Neural Networks. The most important distinguishing feature of this solution is support for multi-topology skeletons, which allows generation of animation for multiple skeletons with a single trained model. In addition, it is possible to use this solution directly in Blender software via the extension [AI Animation Bridge](https://github.com/kottajl/blender-plug-in-for-modifying-3D-animations). The repository contains many trained (on LAFAN1, ACCAD, PFNN, UNOC, 100STYLE datasets) models and the results obtained from them. It also presents quite modular and clear model architecture making it straightforward to extend, modify, or use it with new datasets.

## Installation

### To download without all pretrained models (highly advised considering their size)

#### Set git LFS variable for this terminal session

Bash

```
export GIT_LFS_SKIP_SMUDGE=1
```

Windows CMD
```
set GIT_LFS_SKIP_SMUDGE=1
```

Windows PowerShell

```
$Env:GIT_LFS_SKIP_SMUDGE="1"
```

### To clone with reduced git metadata (highly advised considering commit history size)

```
git clone --depth 1 https://github.com/mpedrak/gnn-skeleton-motion-interpolator.git
```

### To download specific model file

```
cd gnn-skeleton-motion-interpolator
git lfs pull -I "checkpoints/model/v_XX_x.pth"
```

## Current best pretrained models

- **v_51_c** for LAFAN1 skeleton motion in-betweening only

- **v_45_b** for other skeletons (particularly ACCAD, PFNN, UNOC and 100STYLE) 

- **v_50_b** for slightly faster but worse results than v_45_b 

## Usage

- gnn_skeleton_motion_interpolator.py

    - file that can be loaded in [AI Animation Bridge](https://github.com/kottajl/blender-plug-in-for-modifying-3D-animations) Blender extension

- predict.py \<version\> \<file\> \<gap_start\>

    - file that can be used to predict 1 motion in-betweening in 1 file
    - \<version\> is in format v_XX_x
    - \<file\> is absolute or relative path to single BVH file
    - \<gap_start\> is single value for in-betweening start (counting from 1)
    - result is saved as copy of \<file\> with *_pred* in name

- predict_multiple_files.py \<version\>

    - file that can be used to predict multiple motion in-betweening in multiple files
    - \<version\> is in format v_XX_x
    - main data directory and subdirectories with BVH files are defined in code
    - gap starts are calculated based on used version context lengths
    - results are saved as copies of files with *_pred_multi* in name

- train.py \<version\>

    - file that can be used to train single model based on certain config
    - \<version\> (config) is in format v_XX_x
    - automatically saves model checkpoint, info about used skeletons and all logs

- test.py \<version\>

    - file that can be used to do simple test of model (calculates train loss values on test set)
    - \<version\> is in format v_XX_x
    - prints and automatically saves result

- benchmark.py \<version\>

    - file that can be used to do advanced test of model (calculates L2P, L2Q, NPSS and some other metrics on test set)
    - \<version\> is in format v_XX_x
    - prints and automatically saves result

- in *data* directory there are some scripts that can be helpful with datasets handling

## Testing environment

All models were trained and tested on the following hardware and software configuration
- OS: Windows 11 Pro 25H2 (version 26200.8457)
- CPU: Intel Core i5-12600KF 
- GPU: NVIDIA GeForce RTX 5070 Ti
- RAM: 48GB DDR5 4600 MT/s
- Python version: 3.12.10
- Python packages versions

    - bvh: 0.3
    - matplotlib: 3.10.8
    - numpy: 2.3.5
    - PyYAML: 6.0.3
    - scipy: 1.17.1
    - torch: 2.10.0+cu130
    - torch-geometric: 2.7.0
    - tqdm: 4.67.3

## Dataset filtering along with train and test split method

- LAFAN1

    - used: all files 
    - test: subject 5
    - frequency: 30 Hz

- ACCAD

    - used: Male 1, Male 2, Female 1
    - ignored: Male2_B24_WalkToCrouch.bvh, Male2_C19_RunToJumpToWalk.bvh, Male2_C20_RunToPickupBox.bvh
    - test: Male1_A*, Male2_D*, Female1_A* 
    - frequency: 30 Hz

- PFNN

    - used: LocomotionFlat*
    - test: LocomotionFlat11_000.bvh, LocomotionFlat02_001.bvh, LocomotionFlat09_000.bvh
    - frequency: 60 Hz -> 30 Hz

- 100STYLE

    - used: (20 styles) Aeroplane, ArmsBehindBack, Balance, BentForward, BigSteps, BouncyRight, Crouched, CrowdAvoidance, Drunk, FlickLegs, InTheDark, March, Neutral, Proud, Rushed, Skip, StartStop, Tiptoe, TwoFootJump, WildLegs
    - test: *_BR, *_FW
    - frequency: 60 Hz -> 30 Hz

- UNOC

    - used: NO-TABLE* + free_motion_S12.bvh
    - test: *_S2, *_S6, *_S9
    - frequency: 120 Hz -> 30 Hz

- SFU

    - used: all files
    - ignored: 0019_AdvanceBollywoodDance001.bvh, 0019_BasicBollywoodDance001.bvh
    - test: all
    - frequency: 30 Hz
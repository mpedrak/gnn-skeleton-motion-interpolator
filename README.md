### Installation

#### To download without pretrained models

Bash

```
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/mpedrak/gnn-skeleton-motion-interpolator.git
```

Windows CMD
```
set GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 https://github.com/mpedrak/gnn-skeleton-motion-interpolator.git
```

Windows PowerShell

```
$Env:GIT_LFS_SKIP_SMUDGE="1"; git clone --depth 1 https://github.com/mpedrak/gnn-skeleton-motion-interpolator.git
```

#### To download specific model file

```
cd gnn-skeleton-motion-interpolator
git lfs pull -I "checkpoints/model/X_name.pth"
```

#### Current best model

```
6_fk_better.pth
```
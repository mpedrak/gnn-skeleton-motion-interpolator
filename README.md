## Installation

### To download without pretrained models

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

#### Clone with reduced git metadata

```
git clone --depth 1 https://github.com/mpedrak/gnn-skeleton-motion-interpolator.git
```

### To download specific model file

```
cd gnn-skeleton-motion-interpolator
git lfs pull -I "checkpoints/model/v_XX.pth"
```

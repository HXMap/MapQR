# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapQR with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep.py 8
```

Eval MapQR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep.py ./path/to/ckpts.pth 8
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapQR/tools/maptrv2`
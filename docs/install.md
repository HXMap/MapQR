# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n mapqr python=3.8 -y
conda activate mapqr
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install timm.**
```shell
pip install timm
```


**f. Clone MapQR.**
```
git clone git@github.com:HXMap/MapQR.git
```

**g. Install mmdet3d and GKT**
```shell
cd /path/to/MapQR/mmdetection3d
python setup.py develop

cd /path/to/MapQR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

```

**h. Install other requirements.**
```shell
cd /path/to/MapQR
pip install -r requirement.txt
```

**i. Prepare pretrained models.**
```shell
cd /path/to/MapQR
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

### Installed packages for reference.

[pip.txt](https://github.com/user-attachments/files/17054049/pip.txt)

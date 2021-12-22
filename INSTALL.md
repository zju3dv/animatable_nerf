### Set up the python environment

```
conda create -n animatable_nerf python=3.7
conda activate animatable_nerf

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

### Set up datasets

#### Human3.6M dataset

1. Since the license of Human3.6M dataset does not allow us to distribute its data, we cannot release the processed Human3.6M dataset publicly. If someone is interested at the processed data, please email me.
2. Create a soft link:
    ```
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/h36m h36m
    ```

#### ZJU-Mocap dataset

1. Download the ZJU-Mocap dataset from [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/EsCahURVXs1Gm6Oe-sF9zFABWwOLblFd_dP0Ovh8NN2d9w?e=L8P7B0).
2. Create a soft link:
    ```
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```

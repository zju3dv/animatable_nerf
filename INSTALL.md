### Set up the python environment

```
conda create -n animatable_nerf python=3.7
conda activate animatable_nerf

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

If someone wants to run the baseline methods `NHR` and `NT`, please install the libraries:

```
# install poinetnet2
ROOT=/path/to/animatable_nerf
cd $ROOT/lib/csrc/pointnet2
python setup.py build_ext --inplace

# install PCPR
cd ~
git clone https://github.com/wuminye/PCPR.git
cd PCPR
python setup.py install

# install pytorch3d
cd ~
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout tags/v0.4.0
python setup.py install
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
3. If someone wants to run the baseline method `NT`, please run the script:
    ```
    python tools/render_h36m_uvmaps_pytorch3d.py
    ```

#### ZJU-Mocap dataset

1. If someone wants to download the ZJU-Mocap dataset, please fill in the [agreement](https://zjueducn-my.sharepoint.com/:b:/g/personal/pengsida_zju_edu_cn/EUPiybrcFeNEhdQROx4-LNEBm4lzLxDwkk1SBcNWFgeplA?e=BGDiQh), and email me (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.
2. Create a soft link:
    ```
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```

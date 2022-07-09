### Set up the python environment

```shell
conda create -n animatable_nerf python=3.7
conda activate animatable_nerf

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

If someone wants to run the baseline methods `NHR` and `NT`, please install the libraries:

```shell
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

    ```shell
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/h36m h36m
    ```

3. If someone wants to run the baseline method `NT`, please run the script:

    ```shell
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

#### MonoCap Dataset

1. MonoCap is a dataset composed by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/) from [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/).
2. Since the license of [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/) dataset does not allow us to distribute its data, we cannot release the processed MonoCap dataset publicly. If you are interested in the processed data, please download the raw data from [here](https://gvv-assets.mpi-inf.mpg.de/) and email me for instructions on how to process the data.
3. Create a soft link:

    ```shell
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/monocap monocap
    ```

#### SyntheticHuman Dataset

1. SyntheticHuman is a dataset created by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/). It contains multi-view videos of 3D human rendered from characters in the [RenderPeople](https://renderpeople.com/) dataset along with the groud truth 3D models.
2. Since the license of the [RenderPeople](https://renderpeople.com/) dataset does not allow distribution of the 3D models, we cannot realease the processed SyntheticHuman dataset publicly. If you are interested in this dataset, please email me for instructions on how to generate the data.
3. Create a soft link:

    ```shell
    ROOT=/path/to/animatable_nerf
    cd $ROOT/data
    ln -s /path/to/synthetichuman synthetichuman
    ```

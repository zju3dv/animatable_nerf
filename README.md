**News**

* `01/12/2024` [Animatable Neural Fields](https://arxiv.org/abs/2203.08133) gets accepted to TPAMI.
* `07/09/2022` This repository includes the implementation of Animatable SDF (now dubbed [Animatable Neural Fields](https://arxiv.org/abs/2203.08133)).
* `07/09/2022` We release the [extended version](https://arxiv.org/abs/2203.08133) of Animatable NeRF. We evaluated three different versions of Animatable Neural Fields, including vanilla Animatable NeRF, a version where the neural blend weight field is replaced with displacement field and a version where the canonical NeRF model is replaced with a neural surface field (output is SDF instead of volume density, also using displacement field). We also provide evaluation framework for reconstruction quality comparison.
* `10/28/2021` To make the comparison with Animatable NeRF easier on the Human3.6M dataset, we save the quantitative results at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/EpW0AHZh1OtDoa-vTaaCAYgBddyACEICg-941VYgyASk7g?e=W4KvSK), which also contains the results of other methods, including Neural Body, D-NeRF, Multi-view Neural Human Rendering, and Deferred Neural Human Rendering.

# Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies

### [Project Page](https://zju3dv.github.io/animatable_nerf) | [Video](https://www.youtube.com/watch?v=eWOSWbmfJo4) | [Paper](https://arxiv.org/abs/2105.02872) | [Data](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#zju-mocap-dataset) | [Extension](https://arxiv.org/abs/2203.08133)

![teaser](https://zju3dv.github.io/animatable_nerf/images/github_teaser.gif)

> [Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies](https://arxiv.org/abs/2105.02872)  
> Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Xiaowei Zhou, Hujun Bao  
> ICCV 2021

Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

## Run the code on Human3.6M

Since the license of Human3.6M dataset does not allow us to distribute its data, we cannot release the processed Human3.6M dataset publicly. If someone is interested at the processed data, please email me.

We provide the pretrained models at [here](https://drive.google.com/drive/folders/1XH5zUMkguUW64GKulWTo8oOWZra6Dnzy?usp=sharing).

### Test on Human3.6M

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `S9` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_s9p_full/latest.pth`.
2. Test on training human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True
    ```

3. Test on unseen human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume True aninerf_animation True init_aninerf aninerf_s9p test_novel_pose True
    ```

### Visualization on Human3.6M

Take the visualization on `S9` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_s9p_full/latest.pth`.
2. Visualization:
    * Visualize novel views of the 0-th frame

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_novel_view True begin_ith_frame 0
    ```

    * Visualize views of dynamic humans with 3-th camera

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_pose_sequence True test_view "3,"
    ```

    * Visualize mesh

    ```shell
    # generate meshes
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p vis_posed_mesh True
    ```

3. The results of visualization are located at `$ROOT/data/novel_view/aninerf_s9p` and `$ROOT/data/novel_pose/aninerf_s9p`.

### Training on Human3.6M

Take the training on `S9` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:

    ```shell
    # training
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False

    # training the blend weight fields of unseen human poses
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p
    ```

2. Tensorboard:

    ```shell
    tensorboard --logdir data/record/deform
    ```

## Run the code on ZJU-MoCap

If someone wants to download the ZJU-Mocap dataset, please fill in the [agreement](https://pengsida.net/project_page_assets/files/ZJU-MoCap_Agreement.pdf), and email me (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.

We provide the pretrained models at [here](https://drive.google.com/drive/folders/1XH5zUMkguUW64GKulWTo8oOWZra6Dnzy?usp=sharing).

### Test on ZJU-MoCap

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `313` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_313/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_313_full/latest.pth`.
2. Test on training human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True
    ```

3. Test on unseen human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume True aninerf_animation True init_aninerf aninerf_313 test_novel_pose True
    ```

### Visualization on ZJU-MoCap

Take the visualization on `313` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_313/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_313_full/latest.pth`.
2. Visualization:
    * Visualize novel views of the 0-th frame

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True vis_novel_view True begin_ith_frame 0
    ```

    * Visualize views of dynamic humans with 0-th camera

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True vis_pose_sequence True test_view "0,"
    ```

    * Visualize mesh

    ```shell
    # generate meshes
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 vis_posed_mesh True
    ```

3. The results of visualization are located at `$ROOT/data/novel_view/aninerf_313` and `$ROOT/data/novel_pose/aninerf_313`.

### Training on ZJU-MoCap

Take the training on `313` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:

    ```shell
    # training
    python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume False

    # training the blend weight fields of unseen human poses
    python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume False aninerf_animation True init_aninerf aninerf_313
    ```

2. Tensorboard:

    ```shell
    tensorboard --logdir data/record/deform
    ```

## Extended Version

Addtional training and test commandlines are recorded in [train.sh](train.sh) and [test.sh](test.sh).

Moreover, we compiled a list of all possible commands to run in [extension.sh](extension.sh) using on the S9 sequence of the Human3.6M dataset.

This include training, evaluating and visualizing the original Animatable NeRF implementation and all three extented versions.

Here we list the portion of the commands for the SDF-PDF configuration:

```shell
# extension: anisdf_pdf

# evaluating on training poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True

# evaluating on novel poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True test_novel_pose True

# visualizing novel view of 0th frame for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p vis_posed_mesh True

# training base model for anisdf_pdf
python train_net.py --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume False
```

To run Animatable NeRF on other officially supported datasets, simply change the `--cfg_file` and `exp_name` parameters.

Note that for Animatable NeRF with pose-dependent displacement field (NeRF-PDF) and Animatable Neural Surface with pose-dependent displacement field (SDF-PDF), there's no need for training the blend weight fields of unseen human poses.

### MonoCap dataset

MonoCap is a dataset composed by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/) from [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/).

Since the license of [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/) dataset does not allow us to distribute its data, we cannot release the processed MonoCap dataset publicly. If you are interested in the processed data, please download the raw data from [here](https://gvv-assets.mpi-inf.mpg.de/) and email me for instructions on how to process the data.

### SyntheticHuman Dataset

SyntheticHuman is a dataset created by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/). It contains multi-view videos of 3D human rendered from characters in the [RenderPeople](https://renderpeople.com/) dataset along with the groud truth 3D model.

Since the license of the [RenderPeople](https://renderpeople.com/) dataset does not allow distribution of the 3D model, we cannot realease the processed SyntheticHuman dataset publicly. If you are interested in this dataset, please email me for instructions on how to generate the data.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{peng2021animatable,
  title={Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies},
  author={Peng, Sida and Dong, Junting and Wang, Qianqian and Zhang, Shangzhan and Shuai, Qing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={ICCV},
  year={2021}
}
```

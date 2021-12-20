**News**

* `10/28/2021` To make the comparison with Animatable NeRF easier on the Human3.6M dataset, we save the quantitative results at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EbpGvw8LvnFHislCMRzlpLkBNcIdJeWuc96TOJhQ1gX7cQ?e=nirNmn), which also contains the results of other methods, including Neural Body, D-NeRF, Multi-view Neural Human Rendering, and Deferred Neural Human Rendering.

# Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies
### [Project Page](https://zju3dv.github.io/animatable_nerf) | [Video](https://www.youtube.com/watch?v=eWOSWbmfJo4) | [Paper](https://arxiv.org/abs/2105.02872) | [Data]()

![teaser](https://zju3dv.github.io/animatable_nerf/images/github_teaser.gif)

> [Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies](https://arxiv.org/abs/2105.02872)  
> Sida Peng, Junting Dong, Qianqian Wang, Shangzhan Zhang, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> ICCV 2021

Any questions or discussions are welcomed!

**I will update the code and the document.**

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

## Run the code on Human3.6M

Since the license of Human3.6M dataset does not allow us to distribute its data, we cannot release the processed Human3.6M dataset publicly. If someone is interested at the processed data, please email me.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Et7h-48T0_xGtjNGXHwD1-gBPUNJZqd9VPTnsQlkSLktOw?e=TyCnuY).

### Test on Human3.6M

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `S9` as an example.

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth`.
2. Test on training human poses:
    ```
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True
    ```
3. Test on unseen human poses:
    ```
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume True aninerf_animation True init_aninerf aninerf_s9p test_novel_pose True
    ```

### Visualization on Human3.6M

Take the visualization on `S9` as an example. The command lines for visualization are recorded in [visualize.sh](visualize.sh).

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth`.
2. Visualization:
    * Visualize novel views of the 0-th frame
    ```
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_novel_view True begin_ith_frame 0
    ```

    * Visualize views of dynamic humans with 3-th camera
    ```
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_pose_sequence True test_view "3,"
    ```

    * Visualize mesh
    ```
    # generate meshes
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p vis_posed_mesh True
    ```

3. The results of visualization are located at `$ROOT/data/novel_view/aninerf_s9p` and `$ROOT/data/novel_pose/aninerf_s9p`.

### Training on Human3.6M

Take the training on `S9` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:
    ```
    # training
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False

    # training the blend weight fields of unseen human poses
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p
    ```
2. Tensorboard:
    ```
    tensorboard --logdir data/record/deform
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2021animatable,
  title={Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies},
  author={Peng, Sida and Dong, Junting and Wang, Qianqian and Zhang, Shangzhan and Shuai, Qing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={ICCV},
  year={2021}
}
```

from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os
import trimesh
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = 'data/animation/{}'.format(cfg.exp_name)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize(self, output, batch):
        if cfg.vis_tpose_mesh:
            mesh = trimesh.Trimesh(output['vertex'],
                                   output['triangle'],
                                   process=False)
        else:
            mesh = trimesh.Trimesh(output['posed_vertex'],
                                   output['triangle'],
                                   process=False)
        # mesh.show()

        result_dir = 'data/animation/{}'.format(cfg.exp_name)
        os.system('mkdir -p {}'.format(result_dir))
        result_path = os.path.join(result_dir, 'tpose_mesh.npy')
        mesh_path = os.path.join(result_dir, 'tpose_mesh.ply')

        if cfg.vis_posed_mesh:
            result_dir = os.path.join(result_dir, 'posed_mesh')
            os.system('mkdir -p {}'.format(result_dir))
            frame_index = batch['frame_index'][0].item()
            result_path = os.path.join(result_dir,
                                       '{:04d}.npy'.format(frame_index))
            mesh_path = os.path.join(result_dir,
                                     '{:04d}.ply'.format(frame_index))

        np.save(result_path, output)
        mesh.export(mesh_path)

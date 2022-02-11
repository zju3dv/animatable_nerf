import numpy as np
import cv2
import trimesh
import glob
import os
import sys
import open3d as o3d
import pyrender
import pickle


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_colored_pc(pts, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    colors = np.zeros_like(pts)
    colors += rgb
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    return transforms


def calTransformation(v_i, v_j, r, adaptr=False, ratio=10):
    """ from to vertices to T

    Arguments:
        v_i {} -- [description]
        v_j {[type]} -- [description]
    """
    xaxis = np.array([1, 0, 0])
    v = (v_i + v_j) / 2
    direc = (v_i - v_j)
    length = np.linalg.norm(direc)
    direc = direc / length
    rotdir = np.cross(xaxis, direc)
    rotdir = rotdir / np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, xaxis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    # set the minimal radius for the finger and face
    shrink = max(length / ratio, 0.005)
    eigval = np.array([[length / 2 / r, 0, 0], [0, shrink, 0], [0, 0, shrink]])
    T = np.eye(4)
    T[:3, :3] = rotmat @ eigval @ rotmat.T
    T[:3, 3] = v
    return T, r, length


class SkelModel:
    def __init__(self) -> None:
        nJoints = 24
        self.nJoints = nJoints

        smpl_path = 'data/light_stage/smplx/smpl/SMPL_NEUTRAL.pkl'
        smpl = read_pickle(smpl_path)
        kintree = smpl['kintree_table'].T[1:]
        self.kintree = kintree[:, [1, 0]]
        faces = np.loadtxt(
            '/mnt/data/home/pengsida/Codes/EasyMocap/easymocap/visualize/sphere_faces_20.txt',
            dtype=np.int)
        self.ori_faces = faces
        self.vertices = np.loadtxt(
            '/mnt/data/home/pengsida/Codes/EasyMocap/easymocap/visualize/sphere_vertices_20.txt'
        )
        # compose faces
        faces_all = []
        for nj in range(nJoints):
            faces_all.append(faces + nj * self.vertices.shape[0])
        for nk in range(23):
            faces_all.append(faces + nJoints * self.vertices.shape[0] +
                             nk * self.vertices.shape[0])
        self.faces = np.vstack(faces_all)

    def __call__(self, keypoints3d):
        vertices_all = []
        r = 0.02
        # joints
        for nj in range(self.nJoints):
            if nj > 25:
                r_ = r * 0.4
            else:
                r_ = r
            # if keypoints3d[nj, -1] < 0.01:
            #     vertices_all.append(self.vertices * 0.001)
            #     continue
            vertices_all.append(self.vertices * r_ +
                                keypoints3d[nj:nj + 1, :3])
        # limb
        for nk, (i, j) in enumerate(self.kintree):
            # if keypoints3d[i][-1] < 0.1 or keypoints3d[j][-1] < 0.1:
            #     vertices_all.append(self.vertices * 0.001)
            #     continue
            T, _, length = calTransformation(keypoints3d[i, :3],
                                             keypoints3d[j, :3],
                                             r=1)
            if length > 2:  # 超过两米的
                vertices_all.append(self.vertices * 0.001)
                continue
            vertices = self.vertices @ T[:3, :3].T + T[:3, 3:].T
            vertices_all.append(vertices)
        vertices = np.vstack(vertices_all)
        return vertices[:, :]


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


class Renderer(object):
    def __init__(self, focal_length=1000, height=512, width=512):
        self.height = height
        self.width = width
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.focal_length = focal_length

    def render(self, mesh, K, R, T, return_depth=False):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])

        self.renderer.viewport_height = self.height
        self.renderer.viewport_width = self.width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0],
                                                  fy=K[1, 1],
                                                  cx=K[0, 2],
                                                  cy=K[1, 2])
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)

        vertices = mesh.vertices
        # vertices = vertices * 0.005
        # i = int(os.path.basename(mesh_path)[:-4])
        # vertices = vertices + np.load('bounds.npy')[i][0]

        vertices = vertices @ R.T + T
        mesh.vertices = vertices

        mesh.apply_transform(rot)
        normals = compute_normal(mesh.vertices, mesh.faces)
        colors = ((0.5 * normals + 0.5) * 255).astype(np.uint8)
        mesh.visual.vertex_colors[:, :3] = colors

        trans = [0, 0, 0]

        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=colors[n % len(colors)])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh, 'mesh')

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)

        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(
            scene, flags=pyrender.RenderFlags.VERTEX_NORMALS)
        color = color.astype(np.uint8)
        color = color[..., [2, 1, 0]]

        # msk = (rend_depth != 0).astype(np.uint8)
        # color[msk == 0] = 255
        # msk[msk == 1] = 255
        # color = np.concatenate([color, msk[..., None]], axis=2)

        path = 'tmp'
        os.system('mkdir -p {}'.format(path))
        cv2.imwrite('{}/{}.png'.format(path, i), color)


body_model = SkelModel()

joints = np.load('data/h36m/S9/Posing/joints.npy')
parents = np.load('data/h36m/S9/Posing/parents.npy')

data_root = 'data/h36m/S9/Posing'
annots_path = os.path.join(data_root, 'annots.npy')
annots = np.load(annots_path, allow_pickle=True).item()
cameras = annots['cams']
ims = annots['ims']

Ks = np.array(cameras['K'])
Rs = np.array(cameras['R'])
Ts = np.array(cameras['T']).transpose(0, 2, 1) / 1000
Ds = np.array(cameras['D'])

H, W = 1002, 1000

renderer = Renderer(height=H, width=W)

for i in range(len(ims)):
    params_path = os.path.join(data_root, 'new_params/{}.npy'.format(i))
    params = np.load(params_path, allow_pickle=True).item()
    Rh = params['Rh'].astype(np.float32)
    Th = params['Th'].astype(np.float32)
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)
    posed_joints = np.dot(posed_joints, R.T) + Th

    vertices = body_model(posed_joints)
    mesh = trimesh.Trimesh(vertices, body_model.faces)
    cam_ind = 3

    renderer.render(mesh, Ks[cam_ind], Rs[cam_ind], Ts[cam_ind])

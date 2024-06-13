import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
from torch._six import container_abcs
# import collections.abc as container_abcs
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points
from graspnetAPI.utils.utils import xmlReader,parse_posevector


class GraspNetDataset_fusion(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True,voxel_size = 0.005, use_fine = False):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.voxel_size = voxel_size

        if split == 'train':
            self.sceneIds = list(range(0,30))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.pcdpath = []
        self.labelpath = []
        self.sampath = []
        self.scenename = []
        self.inspath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            if use_fine:
                self.pcdpath.append(os.path.join(root, 'fusion_scenes_fine', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes_fine', x, camera, 'seg.npy'))
            else:
                self.pcdpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'seg.npy'))
            # self.inspath.append(os.path.join(root, 'insseg_realsense', x[6:]+'.npy'))
            # self.sampath.append(os.path.join(root, 'sam_fusion', x[6:] + '.npy'))
            self.scenename.append(x.strip())
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.pcdpath)

    def augment_data(self, point_clouds,normals, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        # if np.random.random() > 0.5:
        #     flip_mat = np.array([[-1, 0, 0],
        #                          [0, 1, 0],
        #                          [0, 0, 1]])
        #     point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
        #     normals = transform_point_cloud(normals, flip_mat, '3x3')
        #     for i in range(len(object_poses_list)):
        #         object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
        #     aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        normals = transform_point_cloud(normals, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds,normals, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        fusion_data = np.load(self.pcdpath[index],allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        normal = np.array(fusion_data['normal'])
        color = np.array(fusion_data['color'])
        # seg = np.array(np.load(self.inspath[index]))
        # seg = np.array(np.load(self.sampath[index]))
        seg = np.array(np.load(self.labelpath[index]))
        scene = self.scenename[index]
        if return_raw_cloud:
            return point_cloud, seg

        if self.camera == "kinect":
            mask_x = ((point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] <0.5))
            mask_y = ((point_cloud[:, 1] > -0.5) & (point_cloud[:, 1] < 0.5))
            mask_z = ((point_cloud[:, 2] > -0.02) & (point_cloud[:, 2] < 0.2))
            workspace_mask = (mask_x & mask_y & mask_z)
            point_cloud = point_cloud[workspace_mask]
            normal = normal[workspace_mask]
            color = color[workspace_mask]
            seg = seg[workspace_mask]

        # sample points
        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = point_cloud[idxs]
        seg_sampled = seg[idxs]
        normal_sampled = normal[idxs]
        color_sampled = color[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = normal_sampled.astype(np.float32)
        ret_dict['pcd_color'] = np.concatenate([cloud_sampled.astype(np.float32), color_sampled.astype(np.float32)],
                                               axis=1)

        ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def get_data_label(self, index):
        fusion_data = np.load(self.pcdpath[index], allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        normal = np.array(fusion_data['normal'])
        color = np.array(fusion_data['color'])
        seg = np.array(np.load(self.labelpath[index]))
        scene = self.scenename[index]
        # sample points
        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = point_cloud[idxs]
        seg_sampled = seg[idxs]
        normal_sampled = normal[idxs]
        color_sampled = color[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1


        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        ret_obj_list = []


        # get object poses
        align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))

        scene_reader = xmlReader(
            os.path.join(self.root, 'scenes', scene, self.camera, 'annotations', '0000.xml'))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        poses = []
        for posevector in posevectors:
            obj_idx, pose = parse_posevector(posevector)
            pose = np.matmul(align_mat,pose)
            poses.append(pose)
            obj_list.append(obj_idx+1)
        poses = np.asarray(poses).astype(np.float32)

        for i, obj_idx in enumerate(obj_list):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[i, :3, :4])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[i, :3, :4], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])

            ret_obj_list.append(np.asarray([obj_idx - 1]).astype(np.int64))

            scores = scores[idxs].copy()
            collision = collision[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)


        ret_dict = {}
        if self.augment:
            cloud_sampled,normal_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled,normal_sampled, object_poses_list)
            # ret_dict['aug_trans'] = aug_trans

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = np.ones_like(cloud_sampled).astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['instance_mask'] = seg_sampled
        ret_dict['obj_list'] = ret_obj_list #np.asarray(obj_list).astype(np.int64)
        return ret_dict

def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1)  # here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                               label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels

'''
from https://github.com/rhett-chen/graspness_implementation/blob/main/dataset/graspnet_dataset.py
'''
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)
    return res


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


'''
Author: Haoxiang Ma
'''
import math

import torch
import open3d as o3d
import os
import numpy as np
import sys
import torch.utils.dlpack
import open3d.core as o3c
import torch.nn.functional as F
from tqdm import tqdm
import copy
from SDF import SignedDistanceField
from loss_utils import transform_point_cloud
import matplotlib.pyplot as plt


def load_meshes_labels(root):
    print("start load meshes")
    for i in tqdm(train_objects):
        mesh_path = os.path.join(root, 'models', '%03d' % i, 'nontextured.ply')
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        out_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=2000)
        del mesh
        mesh_dict[i] = out_mesh
    return


def load_SDF(root):
    print("start load object SDFs")
    for i in tqdm(range(88)):
        sdf_path = os.path.join(root, 'models', '%03d' % i, 'grid_sampled_sdf.npz')
        sdf = SignedDistanceField(i, sdf_path)
        sdf_dict[i] = sdf
    return


train_objects = [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 27, 29, 30, 34, 36, 37, 38, 40, 41,
                 43, 44, 46, 48, 51, 52, 56, 57, 58, 60, 61, 62, 63, 66, 69, 70]
novel_objects = [3, 4, 6, 17, 19, 20, 24, 26, 27, 28, 30, 31, 32, 33, 35, 45, 47, 49, 51, 55, 62, 63, 65, 66,
                 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]
mesh_dict = {}
sdf_dict = {}
load_SDF("/home/LAB/r-yanghongyu/data/graspnet")


def contact_point_loss_sdf(end_points):
    left_contact, right_contact, mask, end_points = output_to_grasp(end_points)
    batch_size = left_contact.shape[0]
    instance_mask = end_points['instance_mask']
    fp2_inds = end_points['fp2_inds'].long()
    instance_mask_ = torch.gather(instance_mask, 1, fp2_inds)
    poses = end_points['object_poses_list']
    obj_list = end_points['obj_list']
    width = end_points['grasp_width_fuse']
    grasp_score = end_points['grasp_score_depth']
    contact_collision_loss = 0
    contact_normal_loss = 0
    surface_dist_loss = 0
    width_regularization_loss = 0

    for i in range(batch_size):
        b_instance_mask = instance_mask_[i]
        b_poses = poses[i]
        b_obj_list = obj_list[i]
        _loss_mask = mask[i]
        b_grasp_score = grasp_score[i]
        b_left_contact = left_contact[i]
        b_right_contact = right_contact[i]
        l_normal = torch.zeros_like(b_left_contact)
        l_sdf = torch.zeros(b_left_contact.shape[0]).cuda()
        r_normal = torch.zeros_like(b_right_contact)
        r_sdf = torch.zeros(b_right_contact.shape[0]).cuda()
        for j in range(len(b_obj_list)):
            obj_idx = int(b_obj_list[j])
            obj_mask = (b_instance_mask == (obj_idx + 1))
            obj_pose = b_poses[j]
            obj_sdf = sdf_dict[obj_idx]
            obj_sdf.set_pose(obj_pose)

            r_point_obj = b_right_contact[obj_mask]
            r_point_obj_ = obj_sdf.pos_world2object(r_point_obj)
            r_sdf_obj, r_surface_normal_obj_ = obj_sdf.sample_sdf_and_normal(r_point_obj_)
            r_surface_normal_obj = obj_sdf.normal_object2world(r_surface_normal_obj_)

            l_point_obj = b_left_contact[obj_mask]
            l_point_obj_ = obj_sdf.pos_world2object(l_point_obj)
            l_sdf_obj, l_surface_normal_obj_ = obj_sdf.sample_sdf_and_normal(l_point_obj_)
            l_surface_normal_obj = obj_sdf.normal_object2world(l_surface_normal_obj_)

            l_sdf[obj_mask] = l_sdf_obj
            l_normal[obj_mask] = l_surface_normal_obj
            r_sdf[obj_mask] = r_sdf_obj
            r_normal[obj_mask] = r_surface_normal_obj

        loss_mask = _loss_mask*b_grasp_score
        l_collision_loss = torch.sum(F.relu(0.005-l_sdf) * loss_mask) / (loss_mask.sum() + 1e-5)
        surface_thresh = 0.02
        l_surface_dist_constrain_loss = torch.sum(F.relu(l_sdf - surface_thresh) * loss_mask) / (loss_mask.sum() + 1e-5)

        r_collision_loss = torch.sum(F.relu(0.005-r_sdf) * loss_mask) / (loss_mask.sum() + 1e-5)

        r_surface_dist_constrain_loss = torch.sum(F.relu(r_sdf - surface_thresh) * loss_mask) / (loss_mask.sum() + 1e-5)

        r_normal_contact_loss = torch.sum(
            loss_mask * 0.5 * (
                        1 - F.cosine_similarity((b_right_contact - b_left_contact), r_normal, dim=1))) / (loss_mask.sum() + 1e-5)

        l_normal_contact_loss = torch.sum(loss_mask * 0.5 * (
                    1 - F.cosine_similarity((b_left_contact - b_right_contact), l_normal, dim=1))) / (
                                            loss_mask.sum() + 1e-5)

        width_regularization_mask = ((l_sdf > 0.005) * (r_sdf > 0.005)) * loss_mask
        width_regularization_loss += torch.sum(F.relu(width[i] - 0.02) * width_regularization_mask) / (
                    width_regularization_mask.sum() + 1e-5)

        contact_collision_loss += (l_collision_loss + r_collision_loss)
        contact_normal_loss += (l_normal_contact_loss + r_normal_contact_loss)
        surface_dist_loss += (l_surface_dist_constrain_loss + r_surface_dist_constrain_loss)

    normal_loss = contact_normal_loss / batch_size
    collision_loss = contact_collision_loss / batch_size
    dist_loss = surface_dist_loss / batch_size
    width_loss = width_regularization_loss / batch_size
    end_points['loss/contact_normal_loss'] = normal_loss
    end_points['loss/contact_collsion_loss'] = collision_loss
    end_points['loss/contact_dist_constrain_loss'] = dist_loss
    end_points['loss/width_regularization_loss'] = width_loss
    loss = normal_loss + dist_loss + collision_loss
    return loss, end_points


def output_to_grasp(end_points):
    grasp_score = end_points['grasp_score_pred']
    grasp_center = end_points['batch_grasp_point']
    approaching = -end_points['grasp_top_view_xyz']
    grasp_angle = end_points['grasp_angle_value_pred']
    grasp_width = end_points['grasp_width_pred']
    # one-hot version
    grasp_depth_class = torch.argmax(grasp_score, 2, keepdims=True)
    grasp_depth = ((grasp_depth_class.float() + 1) * 0.01).squeeze()
    grasp_angle = torch.gather(grasp_angle, 2, grasp_depth_class).squeeze()
    grasp_width = torch.gather(grasp_width, 2, grasp_depth_class).squeeze()
    grasp_score = torch.gather(grasp_score, 2, grasp_depth_class).squeeze()
    label_mask, _ = torch.max(end_points['label_mask'], dim=-1)
    end_points['grasp_width_fuse'] = grasp_width

    #normalize grasp_score
    grasp_score_norm = (grasp_score-grasp_score.min())/(grasp_score.max()-grasp_score.min())

    end_points['grasp_score_depth'] = grasp_score_norm
    left_contact, right_contact = differentiable_center_to_contact(grasp_center, approaching,
                                                                   grasp_angle, grasp_width, grasp_depth)
    return left_contact, right_contact, label_mask, end_points


def differentiable_center_to_contact(grasp_center, approaching, inplane_angle, grasp_width, grasp_depth):
    '''
    input:
        grasp_center: b,n,3
        approaching: b,n,3
        inplane_angle: b,n
        grasp_width: b,n
        grasp_depth: b,n (note: base_depth = 0.02 grasp_depth = predict_depth + base_depth)

    :return:
        left_contact: b,n,3
        right_contact: b,n,3
    '''
    batch_size, n, _ = grasp_center.shape
    grasp_center = grasp_center.view(batch_size * n, 3)
    # approaching = approaching.view(-1,3,3)
    approaching = approaching.reshape(-1, 3)
    inplane_angle = inplane_angle.view(-1)

    grasp_width = grasp_width.unsqueeze(2).reshape(-1, 1)
    grasp_depth = grasp_depth.unsqueeze(2).reshape(-1, 1)

    rotation_matrix = batch_viewpoint_params_to_matrix(approaching, inplane_angle)  # b*n,3,3
    height = 0.004
    left_point = torch.cat([grasp_depth - height / 2, -grasp_width / 2, torch.zeros_like(grasp_width)],
                           dim=1).unsqueeze(2)  # b*n,3,1
    right_point = torch.cat([grasp_depth - height / 2, grasp_width / 2, torch.zeros_like(grasp_width)],
                            dim=1).unsqueeze(2)  # b*n,3,1
    left_contact = torch.matmul(rotation_matrix, left_point).squeeze() + grasp_center  # b*n,3
    right_contact = torch.matmul(rotation_matrix, right_point).squeeze() + grasp_center  # b*n,3
    left_contact = left_contact.view(batch_size, n, 3)
    right_contact = right_contact.view(batch_size, n, 3)
    return left_contact, right_contact


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch

        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y, 1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix
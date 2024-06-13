""" Modules for GraspNet baseline model.
    Author: chenxi-wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup, OnlyGroup  # , BallRotQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix,generate_lalo_view8
from pointnet2_utils import furthest_point_sample
from SDF import SignedDistanceField
from label_generation import process_graspness
from pointnet2_util import *
import open3d as o3d




class ApproachNet_regression_view_fps(nn.Module):
    def __init__(self, num_view, seed_feature_dim):

        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 3, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.graspable_head = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, 3, 1),
        )

    def forward(self, seed_xyz, seed_features, end_points, is_training=False,sample_from_sp_distribution=False):

        B, _, _ = seed_xyz.size()
        end_points['fp2_xyz'] = seed_xyz
        graspable = self.graspable_head(seed_features)
        objectness_score = graspable[:, :2]
        graspness_score = graspable[:, 2]

        end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        if is_training:
            end_points = process_graspness(end_points)

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = (graspness_score > 0.1) & objectness_mask
        graspable_inds_list = []
        for i in range(B):
            graspable_points = seed_xyz[i][graspness_mask[i] == 1]
            sample_inds = furthest_point_sample(graspable_points.unsqueeze(0), 1024).long()
            inds = torch.where(graspness_mask[i] == 1)[0].unsqueeze(0)
            graspable_inds = torch.gather(inds, 1, sample_inds)
            graspable_inds_list.append(graspable_inds)
        graspable_inds = torch.cat(graspable_inds_list, dim=0)
        graspable_xyz = torch.gather(seed_xyz, 1, graspable_inds.unsqueeze(2).repeat(1, 1, 3))
        graspable_features = torch.gather(seed_features.permute(0, 2, 1), 1,
                                          graspable_inds.unsqueeze(2).repeat(1, 1, 256))
        graspable_features = graspable_features.permute(0, 2, 1)
        _, num_seed, _ = graspable_xyz.size()

        end_points['fp2_xyz'] = graspable_xyz
        end_points['fp2_inds'] = graspable_inds
        end_points['fp2_features'] = graspable_features
        fp2_graspness = torch.gather(graspness_score, 1, graspable_inds)
        end_points['fp2_graspness'] = fp2_graspness

        features = F.relu(self.bn1(self.conv1(graspable_features)), inplace=True)
        features = self.conv2(features)

        vp_xyz = features.transpose(1, 2).contiguous()  # (B, num_seed, 3)
        end_points['view_prediction'] = vp_xyz

        template_views = generate_grasp_views(300).to(seed_features.device)  # (num_view, 3)
        template_views = template_views.view(1, 1, 300, 3).expand(B, num_seed, -1, -1).contiguous()  # (B, num_seed, num_view, 3)

        top_view_inds = torch.argmax(torch.cosine_similarity(template_views, vp_xyz.unsqueeze(2), dim=-1), dim=2)
        vp_xyz_ = vp_xyz.view(-1, 3)

        # no rotation here
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # transfer approach to 3x3
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot
        return end_points


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]

        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot, features=None, ret_points = False):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors
                feature: [torch.FloatTensor, (batch_size,c, num_seed)]
                    feature such as normal

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot, features
            ))  # (batch_size, feature_dim, num_seed, nsample)
        grouped_features = torch.stack(grouped_features,
                                       dim=3)  # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed * num_depth,
                                                 self.nsample)  # (batch_size, feature_dim, num_seed*num_depth, nsample)
        if ret_points:
            return  grouped_features.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.nsample, self.in_dim)
        vp_features = self.mlps(
            grouped_features
        )  # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        )  # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features

class OperationNet_regression(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 4, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()

        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_score_pred'] = vp_features[:, 0]
        # eps = 1e-8
        end_points['grasp_angle_pred'] = F.tanh(vp_features[:, 1:3])
        sin2theta = F.tanh(vp_features[:, 1])
        cos2theta = F.tanh(vp_features[:, 2])
        angle = 0.5 * torch.atan2(sin2theta, cos2theta)
        angle[angle < 0] += np.pi
        # sin2theta = torch.clamp(vp_features[:, 1], min=-1.0+eps, max=1.0-eps)
        # cos2theta = torch.clamp(vp_features[:, 1], min=-1.0 + eps, max=1.0 - eps)
        end_points['grasp_angle_value_pred'] = angle
        end_points['grasp_width_pred'] = F.sigmoid(vp_features[:, 3]) * 0.1
        return end_points


class ToleranceNet_regression(nn.Module):
    """ Grasp tolerance prediction.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points  # ,vp_features


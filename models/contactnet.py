import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from pointnet2_utils import three_nn,three_interpolate


class ContactNet(nn.Module):
    def __init__(self,is_training = False):
        super().__init__()
        self.obj_encoder = Pointnet2BackboneSeg()
        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(256 + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, 1),
        )
        self.head_cos = nn.Sequential(
            nn.Conv1d(256 + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, 1),
        )
        self.is_training = is_training

    def forward(self, end_points):
        obj_pcd = end_points['point_clouds']
        grasp_pcd = end_points['grasp_points']
        seed_features, seed_xyz, end_points = self.obj_encoder(obj_pcd, end_points)
        grasp_features = self.grasp_encoder(grasp_pcd.permute(0,2,1))
        grasp_embedding = F.max_pool1d(grasp_features,kernel_size = [grasp_features.size(2)])
        dist, idx = three_nn(obj_pcd, seed_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        seed_features = three_interpolate(
            seed_features, idx, weight
        )
        grasp_embedding = grasp_embedding.repeat(1, 1, seed_features.size(2))
        cmap = self.head(torch.cat([seed_features,grasp_embedding],dim=1)).squeeze(1) #(B,N)
        cmap_cos = self.head_cos(torch.cat([seed_features, grasp_embedding], dim=1)).squeeze(1)  # (B,N)

        end_points['cmap'] = cmap
        end_points['cmap_cos'] = cmap_cos
        cmap_gt = end_points['cmap_label']
        criterion = nn.MSELoss(reduction='none')
        weight = cmap_gt
        loss_dist = criterion(cmap, cmap_gt)
        loss_dist = torch.sum(loss_dist * weight, dim=-1) / (torch.sum(weight,dim=-1) + 1e-6)

        cmap_cos_gt = end_points['cmap_label_proj']
        weight_cos = cmap_cos_gt
        criterion2 = nn.MSELoss(reduction='none')
        loss_proj = criterion2(cmap_cos, cmap_cos_gt)
        loss_proj = torch.sum(loss_proj * weight_cos, dim=-1) / (torch.sum(weight_cos, dim=-1) + 1e-6)
        return loss_dist,loss_proj,end_points


class Pointnet2BackboneSeg(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.04,
            nsample=64,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=256,
            radius=0.1,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=64,
            radius=0.3,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        sa1_inds = fps_inds
        sa1_xyz = xyz
        sa1_features = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        sa2_inds = fps_inds
        sa2_xyz = xyz
        sa2_features = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        sa3_xyz = xyz
        sa3_features = features

        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        sa4_xyz = xyz
        sa4_features = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        features = self.fp2(sa2_xyz, sa3_xyz, sa2_features, features)
        features = self.fp3(sa1_xyz, sa2_xyz, sa1_features, features)
        end_points['fp3_features'] = features
        end_points['fp3_xyz'] = sa1_xyz
        num_seed = end_points['fp3_xyz'].shape[1]
        end_points['fp3_inds'] = sa1_inds[:, 0:num_seed]  # indices among the entire input point clouds

        return features, end_points['fp3_xyz'], end_points
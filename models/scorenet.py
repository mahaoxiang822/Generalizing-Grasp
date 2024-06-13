import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math
from backbone import Pointnet2BackboneCls
from loss_utils import transform_point_cloud


class GraspScoreNet(nn.Module):
    def __init__(self, is_training = False):
        super().__init__()
        self.backbone = Pointnet2BackboneCls()
        self.score_head = nn.Sequential(
            nn.Linear(1024+128,512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256,1),
        )
        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
        )
        self.is_training = is_training

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        B,_,_ = pointcloud.shape
        grasp_points = end_points['grasp_points']
        rotations = end_points['rotation'].view(B,3,3)
        translations = end_points['translation'].view(B,3,1)
        transform = torch.cat([rotations,translations],dim=-1)
        ones = grasp_points.new_ones(grasp_points.size()[0:2], device=grasp_points.device).unsqueeze(-1)
        cloud_ = torch.cat([grasp_points, ones], dim=2)
        grasp_points_transformed = torch.matmul(transform, cloud_.permute(0,2,1))
        region_point_cloud = torch.cat([pointcloud,grasp_points_transformed.permute(0,2,1)],dim=1).detach()
        region_feature = self.backbone(region_point_cloud, end_points).squeeze(2)  # (B,1024)
        grasp_pose_feature = self.grasp_encoder(grasp_points_transformed)
        grasp_embedding = F.max_pool1d(grasp_pose_feature, kernel_size=[grasp_pose_feature.size(2)]).squeeze(-1)
        feature = torch.cat([region_feature, grasp_embedding], dim=-1)
        predict_score = self.score_head(feature)
        if self.is_training:
            loss = self.get_loss(predict_score,end_points['score'])
            end_points['loss/score_loss'] = loss
            return loss, end_points
        else:
            return predict_score, end_points

    def get_loss(self,predict_score, target_score):
        criterion = nn.SmoothL1Loss()
        return criterion(predict_score,target_score)
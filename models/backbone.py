""" PointNet2 backbone for feature learning.
    Author: Charles R. Qi
"""
import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

'''
in 3 x N
out 256 x 1024
'''
class Pointnet2Backbone(nn.Module):
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
                npoint=2048,
                radius=0.04,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

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

        xyz, _ = self._break_up_pc(pointcloud)
        # xyz = pointcloud.transpose(1, 2).contiguous()
        features = end_points['feats'].transpose(1, 2).contiguous()
        end_points['input_xyz'] = xyz

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        sa1_inds = fps_inds
        sa1_xyz = xyz
        sa1_features = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        sa2_inds = fps_inds
        sa2_xyz = xyz
        sa2_features = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        sa3_xyz = xyz
        sa3_features = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        sa4_xyz = xyz
        sa4_features = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(sa3_xyz, sa4_xyz, sa3_features, sa4_features)
        features = self.fp2(sa2_xyz, sa3_xyz, sa2_features, features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = sa2_xyz
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = sa1_inds[:,0:num_seed] # indices among the entire input point clouds

        return features, end_points['fp2_xyz'], end_points


class Pointnet2BackboneCls(nn.Module):

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.2,
            nsample=32,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=None,
            radius=0.5,
            nsample=None,
            mlp=[256, 256, 512, 1024],
            use_xyz=True,
            normalize_xyz=True
        )



    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):

        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        sa1_inds = fps_inds
        sa1_xyz = xyz
        sa1_features = features

        xyz, features, fps_inds = self.sa2(xyz, features)
        sa2_inds = fps_inds
        sa2_xyz = xyz
        sa2_features = features

        xyz, features, fps_inds = self.sa3(xyz, features, fps_inds)
        sa3_xyz = xyz
        sa3_features = features
        return features


if __name__ == "__main__":
    inp = torch.rand(2, 1024, 3).cuda().float()
    network = Pointnet2Backbone().cuda()
    out1,out2,out3 = network(inp)
    print(out1.size())


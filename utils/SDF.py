import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loss_utils import transform_point_cloud
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def grads2img(mG):
    import math
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    plt.imshow(mGrgb)
    plt.savefig("grad.png")
    print(mGrgb.shape)
    return

class SignedDistanceField:
    def __init__(self, object_id, path, pose=None):
        self.object_id = object_id
        self.path = path
        self.sdf_data = np.load(path)

        self.pose = pose #(3,4)
        points = torch.from_numpy(self.sdf_data["points"]).float()
        self.sdf = torch.from_numpy(self.sdf_data["sdf"]).float()
        self.min_bound = points[0, 0, 0]
        self.max_bound = points[-1, -1, -1]

    def sample_sdf_and_normal(self, position):
        '''
        position (N,3)
        '''
        points = torch.from_numpy(self.sdf_data["points"]).float().requires_grad_(True).cuda()
        sdf = torch.from_numpy(self.sdf_data["sdf"]).float().requires_grad_(True).cuda()
        min_bound = points[0, 0, 0]
        max_bound = points[-1, -1, -1]
        position.requires_grad_(True)
        n, dim = position.size()
        x_dim, y_dim, z_dim = sdf.size()
        # normalize to [-1,1]
        min_bound = torch.flip(min_bound, dims=[-1])
        max_bound = torch.flip(max_bound, dims=[-1])
        position_ = torch.flip(position,dims=[-1])
        position_grid = ((position_ - min_bound) / (max_bound - min_bound) - 0.5) * 2  # N,3
        position_grid_ = position_grid.view(1, n, 1, 1, dim)
        sdf_value = sdf.view(1, 1, x_dim, y_dim, z_dim)
        sdf_value_sampled = F.grid_sample(sdf_value, position_grid_, mode='bilinear', padding_mode='border')
        sdf_value_sampled = sdf_value_sampled.squeeze() # N,
        normal = torch.autograd.grad(sdf_value_sampled.sum(), position_,retain_graph=True)[0] # N,3
        normal = torch.flip(normal, dims=[-1])
        return sdf_value_sampled, normal

    def pos_world2object(self,pos):
        pos = pos - self.pose[:,3]
        object_frame_pos = torch.matmul(pos,self.pose[:3,:3])
        return object_frame_pos

    def normal_object2world(self,normal):
        world_frame_normal = torch.matmul(self.pose[:3,:3], normal.T).T
        return world_frame_normal

    def set_pose(self,pose):
        self.pose = pose

if __name__ == "__main__":
    import os
    path = "/data/mahaoxiang/graspnet/models/007"
    sdf = SignedDistanceField(0,os.path.join(path,"grid_sampled_sdf.npz"))
    x_linespace = torch.linspace(float(sdf.min_bound[0])-0.01, float(sdf.max_bound[0])+0.01, 6)
    y_linespace = torch.linspace(float(sdf.min_bound[1])-0.01, float(sdf.max_bound[1])+0.01, 6)
    z_linespace = torch.linspace(float(sdf.min_bound[2])-0.01, float(sdf.max_bound[2])+0.01, 6)
    x_grid, y_grid, z_grid = torch.meshgrid(x_linespace, y_linespace, z_linespace)
    x_grid = x_grid.unsqueeze(-1)
    y_grid = y_grid.unsqueeze(-1)
    z_grid = z_grid.unsqueeze(-1)
    grid_3d = torch.cat([z_grid, y_grid, x_grid], dim=-1).view(-1,3).requires_grad_(True)
    # grid_3d = torch.from_numpy(np.load("../test.npy")).requires_grad_(True)
    sdf_value,normal = sdf.sample_sdf_and_normal_test(grid_3d)
    grid_3d = grid_3d.flip(dims=[1])
    normal = normal.flip(dims=[1])
    res = torch.cat([grid_3d,normal],dim=-1).detach().numpy()
    np.save("sdf_val.npy",res)







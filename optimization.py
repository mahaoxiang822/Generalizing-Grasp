import math
import os
import random
import sys
import time

import numpy as np
import open3d as o3d
import argparse
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from graspnetAPI import GraspGroup,Grasp, GraspNetEval
from graspnetAPI.graspnet_eval import *
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from mink_dataset import GraspNetDataset_fusion, minkowski_collate_fn
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--c_checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--s_checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--save_dir', default='log', help='dir save the output grasp from network')
parser.add_argument('--dump_dir', default='log', help='Dump dir to save the refined grasp')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 2]')
parser.add_argument('--num_workers', type=int, default=2, help='workers num during training [default: 2]')

cfgs = parser.parse_args()


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create Dataset and Dataloader
TEST_DATASET = GraspNetDataset_fusion(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera=cfgs.camera,
                                   num_points=20000, remove_outlier=True, augment=False, load_label=False,use_fine=False)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from contactnet import ContactNet
from scorenet import GraspScoreNet

cnet = ContactNet()
snet = GraspScoreNet()

cnet.to(device)
snet.to(device)

c_checkpoint = torch.load(cfgs.c_checkpoint_path)
cnet.load_state_dict(c_checkpoint['model_state_dict'])
s_checkpoint = torch.load(cfgs.s_checkpoint_path)
snet.load_state_dict(s_checkpoint['model_state_dict'])
cnet.eval()
snet.eval()


class CmapEval(GraspNetEval):

    def pc_normalize(self, pc_o, pc_g):
        centroid = torch.mean(pc_o, dim=0)
        pc_o = pc_o - centroid
        pc_g = pc_g - centroid
        return pc_o,pc_g, centroid

    def generate_scene(self,scene_id):

        model_list, dexmodel_list, obj_list = self.get_scene_models(scene_id, ann_id=0)
        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)
        model_trans_list = list()
        _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, 0)
        # transform pose to world coord
        # note: because we use the view 0, only align_mat is needed
        transform_pose_list = []
        for pose in pose_list:
            pose = np.matmul(align_mat, pose)
            transform_pose_list.append(pose)

        seg_mask = list()
        for i, model in enumerate(model_sampled_list):
            model_trans = transform_points(model, transform_pose_list[i])
            seg = (obj_list[i]+1) * np.ones(model_trans.shape[0], dtype=np.int32)
            model_trans_list.append(model_trans)
            seg_mask.append(seg)
        seg_mask = np.concatenate(seg_mask, axis=0)
        scene = np.concatenate(model_trans_list, axis=0)
        return scene,seg_mask

    def opt(self):

        for batch_idx, batch_data in enumerate(TEST_DATALOADER):

            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            point_clouds = batch_data['point_clouds'].squeeze().detach().cpu().numpy()
            seg_mask = batch_data['instance_mask'].squeeze().detach().cpu().numpy()
            N, _ = point_clouds.shape
            gg_file = os.path.join(cfgs.save_dir, SCENE_LIST[batch_idx], cfgs.camera, "result.npy")
            res_file = os.path.join(cfgs.save_dir, SCENE_LIST[batch_idx], cfgs.camera,"save.npy")
            grasp_data = np.load(res_file)
            centers = grasp_data[:,8:11]
            grasp_group = GraspGroup().from_npy(gg_file)

            # todo apply nms on saved data
            grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)
            centers_afternms = grasp_group.grasp_group_array[:, 13:16]
            nms_index = np.argmin(np.sum(np.abs(centers_afternms[:,None]- centers[None]),axis=2),axis=1)
            grasp_data = grasp_data[nms_index]


            centers = grasp_data[:,8:11]
            scene, seg_mask_ = self.generate_scene(int(SCENE_LIST[batch_idx][-3:]))
            indices = compute_closest_points(centers, scene)
            model_to_grasp = seg_mask_[indices]
            model_ids = np.unique(model_to_grasp)
            pred_list = []
            for i in model_ids:
                if i==0:
                    continue
                grasp_i = grasp_data[model_to_grasp == i]
                index = np.argsort(grasp_i[:,0])
                index = index[::-1]
                grasp_i = grasp_i[index]
                grasp_i[:,-1] = i
                pred_list.append(grasp_i[:5])
            pred_list = np.concatenate(pred_list,axis=0)
            approach = pred_list[:,4:7]
            grasp_angle = pred_list[:,7]
            grasp_center = pred_list[:,8:11]
            obj_ids = pred_list[:,11,None]

            rotation_matrix = batch_viewpoint_params_to_matrix(torch.from_numpy(approach).cuda(), torch.from_numpy(grasp_angle).cuda())
            rotation_matrix = rotation_matrix.view(-1, 9).cpu().numpy()

            gg_array = np.concatenate([pred_list[:,0:4], rotation_matrix, grasp_center, obj_ids],axis=-1)
            
            selected_grasp_group = GraspGroup(gg_array)

            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[batch_idx], cfgs.camera)
            grasp_refined_list = []
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(len(selected_grasp_group)):
                cropped_points = point_clouds[seg_mask == selected_grasp_group[i].object_id]
                cropped_points = torch.from_numpy(cropped_points).cuda()
                selected_grasp = copy.deepcopy(selected_grasp_group[i])
                if selected_grasp.score > 1:
                    grasp_refined = copy.deepcopy(selected_grasp_group[i].grasp_array)
                    # np.savez(os.path.join(save_dir, str(i) + ".npz"), grasp=selected_grasp.grasp_array,
                    #          points=cropped_points.detach().cpu().numpy(), grasp_refined=grasp_refined, cmap_before = 0, cmap_end = 0,grasp_score_before = 0, grasp_score = 0)
                    grasp_refined_list.append(grasp_refined)
                    continue
                optimized_grasp_tensor= torch.from_numpy(selected_grasp.grasp_array).cuda().float()
                translation = torch.from_numpy(selected_grasp.translation).cuda().float()
                approaching = torch.from_numpy(approach[i]).cuda().float()
                inplane_angle = torch.from_numpy(grasp_angle[i,None]).cuda().float()
                depth = torch.tensor([selected_grasp.depth]).cuda().float()
                width = torch.tensor([selected_grasp.width]).cuda().float()
                translation = torch.autograd.Variable(translation,requires_grad = True)
                approaching = torch.autograd.Variable(approaching,requires_grad = True)

                value = angle2value(inplane_angle)
                value = torch.autograd.Variable(value,requires_grad = True)
                depth = torch.autograd.Variable(depth,requires_grad = True)
                width = torch.autograd.Variable(width,requires_grad = True)
                optimization_steps = 300
                params_list=[]
                params_list.append(dict(params=translation, lr=0.0002))
                params_list.append(dict(params=approaching, lr=0.002))
                params_list.append(dict(params=value, lr=0.002))
                params_list.append(dict(params=width, lr=0.0001))

                optimizer = torch.optim.Adam(params_list)

                flag = False
                for t in range(optimization_steps):
                    optimizer.zero_grad()
                    if torch.isnan(width):
                        flag = True
                        break
                    inplane_angle = value2angle(value)
                    optimized_grasp_tensor[13:16] = translation
                    rotation = batch_viewpoint_params_to_matrix(approaching.unsqueeze(0),
                                                                inplane_angle).squeeze()

                    optimized_grasp_tensor[4:13] = rotation.view(-1)
                    optimized_grasp_tensor[3] = depth
                    optimized_grasp_tensor[1] = width
                    optimized_grasp = Grasp(optimized_grasp_tensor.detach().cpu().numpy())
                    grasp_mesh_o3d = optimized_grasp.to_open3d_geometry()
                    grasp_points_o3d = grasp_mesh_o3d.sample_points_uniformly(64)
                    grasp_points = np.asarray(grasp_points_o3d.points)
                    grasp_points = torch.from_numpy(grasp_points).cuda()
                    left_contact, right_contact = differentiable_center_to_contact(translation,approaching,
                                                                                        inplane_angle,width,depth)

                    cmap, cmap_proj, loss_distance = calculate_cmap(left_contact, right_contact, cropped_points)

                    norm_cropped_points, norm_grasp_points, obj_center = self.pc_normalize(cropped_points, grasp_points)
                    input = {
                        'point_clouds':norm_cropped_points.unsqueeze(0).float(),
                        'grasp_points':norm_grasp_points.unsqueeze(0).float().detach(),
                        'cmap_label':cmap.unsqueeze(0).float(),
                        'cmap_label_proj': cmap_proj.unsqueeze(0).float(),
                    }

                    loss_cmap_dist, loss_cmap_proj, end_points = cnet(input)
                    loss_cmap = loss_cmap_dist + 0.2 * loss_cmap_proj

                    GRASP = copy.deepcopy(optimized_grasp)
                    GRASP.translation = np.array([0, 0, 0])
                    GRASP.rotation_matrix = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                    GRASP_points = GRASP.to_open3d_geometry().sample_points_uniformly(64)
                    GRASP_points = np.asarray(GRASP_points.points)
                    translation_ = translation-obj_center
                    score_input = {
                        'point_clouds': norm_cropped_points.unsqueeze(0).float(),
                        'grasp_points': torch.from_numpy(GRASP_points).cuda().unsqueeze(0).float().detach(),
                        'rotation': rotation.view(-1).unsqueeze(0).float(),
                        'translation': translation_.unsqueeze(0).float(),
                    }
                    predict_score, _ = snet(score_input)
                    loss_score = np.log(12)-predict_score

                    if t == 0:
                        start_loss_cmap = loss_cmap.item()
                        start_loss_score = loss_score.item()

                    center_distance = F.relu(torch.sqrt(torch.sum((cropped_points - translation.unsqueeze(0)) ** 2, dim=-1)+1e-6)-0.001)
                    loss_center = F.relu(torch.min(center_distance)-0.001)
                    loss = loss_cmap_dist + 0.2 * loss_cmap_proj + 5 * loss_center + loss_distance + 0.1 * loss_score

                    if t%30 ==0:
                        print("step ", t, " loss cmap %f" % loss_cmap_dist.item(), "loss center %f" % loss_center,
                              "loss distance %f" % loss_distance, "loss score %f" % loss_score)

                    loss.backward()
                    optimizer.step()

                grasp_refined = copy.deepcopy(selected_grasp_group[i].grasp_array)
                if ((loss_cmap.item() < start_loss_cmap) and (loss_score.item() < start_loss_score)) or flag:

                    print("refined one grasp: before cmap %f" % start_loss_cmap, " after cmap %f" % loss_cmap.item())
                    grasp_refined[13:16] = translation.detach().cpu().numpy()
                    inplane_angle = value2angle(value)
                    rotation = batch_viewpoint_params_to_matrix(approaching.unsqueeze(0),
                                                     inplane_angle).squeeze()
                    grasp_refined[4:13] = rotation.view(-1).detach().cpu().numpy()
                    grasp_refined[3] = depth.detach().cpu().numpy()
                    grasp_refined[1] = width.detach().cpu().numpy()

                # np.savez(os.path.join(save_dir, str(i) + ".npz"), grasp=selected_grasp.grasp_array,
                #          points=cropped_points.detach().cpu().numpy(), grasp_refined = grasp_refined)
                grasp_refined_list.append(grasp_refined)

            gg_r = GraspGroup(np.vstack(grasp_refined_list))
            save_path = os.path.join(save_dir, 'result.npy')
            gg_r.save_npy(save_path)


def save_grasp(grasp_score, grasp_xyz, approach, angle, depth, width, obj_mask):
    batch_size = len(grasp_xyz)
    grasp_preds = []
    for i in range(batch_size):
        score = grasp_score[i].unsqueeze(1).float()
        grasp_center = grasp_xyz[i].float()
        approaching = -approach[i].float()

        grasp_angle = angle[i].unsqueeze(1).float()
        grasp_width = width[i].unsqueeze(1).float()
        grasp_width = torch.clamp(grasp_width, min=0, max=0.1)
        grasp_depth = depth[i].unsqueeze(1).float()
        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)
        # merge preds
        grasp_height = 0.02 * torch.ones_like(score)
        obj_ids = obj_mask[i].unsqueeze(1).float()
        b_grasp_preds = torch.cat(
            [score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1)
        b_grasp_preds = b_grasp_preds[grasp_score[i] > 1].detach().cpu().numpy()
        gg = GraspGroup(b_grasp_preds)
        grasp_preds.append(gg)
    return grasp_preds


def differentiable_center_to_contact(grasp_center,approaching,inplane_angle, grasp_width, grasp_depth):
    approaching = approaching.unsqueeze(0)
    inplane_angle = inplane_angle
    rotation_matrix = to_matrix(approaching, inplane_angle).squeeze()  # b*n,3,3

    height = 0.004
    left_point = torch.cat([grasp_depth - height / 2, -grasp_width / 2, torch.zeros_like(grasp_width)],
                           dim=0).unsqueeze(1)  # 3,1
    right_point = torch.cat([grasp_depth - height / 2, grasp_width / 2, torch.zeros_like(grasp_width)],
                            dim=0).unsqueeze(1)  # 3,1
    left_contact = torch.matmul(rotation_matrix, left_point).squeeze() + grasp_center  # b*n,3
    right_contact = torch.matmul(rotation_matrix, right_point).squeeze() + grasp_center  # b*n,3
    return left_contact, right_contact



def to_matrix(batch_towards, batch_angle):
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
    proj = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, proj, -sin, zeros, sin, proj], dim=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def calculate_cmap(left_contact, right_contact, point_clouds):
    left_contact = left_contact.unsqueeze(0)
    right_contact = right_contact.unsqueeze(0)
    contacts = torch.cat([left_contact, right_contact], dim=0)  # (2,3)
    distance_ = torch.sqrt(torch.sum((point_clouds[:, None] - contacts) ** 2, dim=-1))
    n, _ = point_clouds.shape
    l2r_ = (left_contact - right_contact).repeat(n, 1)
    distance, distance_index = torch.min(20*distance_, dim=-1)
    direct = torch.ones_like(distance_index)
    direct[distance_index == 0] = -1
    direct = direct.unsqueeze(1)
    distance_index = distance_index.view(-1, 1, 1).repeat(1, 1, 3)
    p2c = torch.gather((point_clouds[:, None] - contacts), 1, distance_index).squeeze()  # n,3
    proj = torch.cosine_similarity(p2c, l2r_ * direct, dim=1)
    distance_sin = torch.sin(torch.acos(proj)) * distance
    cmap_proj = 1 - 2 * (torch.sigmoid(4 * distance_sin) - 0.5)
    cmap = 1 - 2 * (torch.sigmoid(2 * distance) - 0.5)
    min_contact_distance = torch.min(distance_,dim=0)[0]
    loss_contact_distance = F.relu(min_contact_distance-0.01).mean() + 5*F.relu(0.005-min_contact_distance).mean()
    return cmap,cmap_proj,loss_contact_distance


def contact_collision_loss(contacts, point_clouds):
    contacts = torch.vstack(contacts)
    distance_ = torch.sqrt(torch.sum((point_clouds[:, None] - contacts) ** 2, dim=-1)+1e-8)
    min_contact_distance = torch.min(distance_, dim=0)[0]
    loss_contact_distance = F.relu(0.01-min_contact_distance).mean()
    return loss_contact_distance


def angle2value(angle):
    return torch.cat([torch.sin(angle * 2),torch.cos(angle * 2)])


def value2angle(value):
    sin2theta = F.tanh(value[0])
    cos2theta = F.tanh(value[1])
    angle = 0.5 * torch.atan2(sin2theta, cos2theta)
    if angle<0:
        angle+=np.pi
    return angle[None]


if __name__ == '__main__':
    cmap_eval = CmapEval(root=cfgs.dataset_root,camera=cfgs.camera)
    cmap_eval.opt()
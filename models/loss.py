""" Loss functions for training.
    Author: chenxi-wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD, \
    transform_point_cloud, generate_grasp_views, \
    batch_viewpoint_params_to_matrix, huber_loss
import contact_point_loss


def get_loss(end_points):
    objectness_loss, end_points = compute_graspable_loss_sparse(end_points)
    view_loss, end_points = compute_robust_view_loss_regression(end_points)
    grasp_loss, end_points = compute_grasp_loss_regression(end_points)
    contact_sdf_loss, end_points = contact_point_loss.contact_point_loss_sdf(end_points)
    loss = objectness_loss + view_loss + 0.2*grasp_loss + 0.1 * contact_sdf_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_graspable_loss_sparse(end_points):
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score']
    graspness_label = end_points['graspness_label']
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    objectness_loss = criterion1(objectness_score, objectness_label)
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    end_points['loss/stage1_objectness_loss'] = objectness_loss

    loss_mask = end_points['objectness_label'].bool()
    loss = criterion2(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error
    end_points['loss/stage1_graspness_loss'] = loss
    return loss+objectness_loss, end_points

def compute_robust_view_loss_regression(end_points):
    view_direction = end_points['view_prediction']
    template_views = end_points['batch_grasp_view_all']  # (B, Ns,V, 3)
    view_label = end_points['batch_grasp_view_label']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    V = view_label.size(2)
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)

    batch_grasp_label = end_points['batch_grasp_label_all']
    B, Ns, V, A, D = batch_grasp_label.size()
    target_labels = batch_grasp_label.view(B, Ns,V, -1)
    target_labels = target_labels.sum(3)

    max_target_labels, target_inds = torch.max(target_labels, dim=2)
    target_inds = target_inds.view(B, Ns, 1, 1).expand(-1, -1, -1, 3)
    target_direction = torch.gather(template_views, 2, target_inds).squeeze(2)

    # normalize
    min = torch.min(target_labels,dim=2,keepdim=True)[0]
    max = torch.max(target_labels,dim=2,keepdim=True)[0]
    target_labels_norm = (target_labels-min)/(max-min+1e-5)
    neighbor_view = (F.cosine_similarity(view_direction.unsqueeze(2),template_views,dim=-1)+1)/2
    neighbor_view[neighbor_view<np.cos(np.pi/6)] = 0
    predict_view_score = torch.sum(neighbor_view*target_labels_norm,dim=-1)/(neighbor_view.sum(dim=-1))
    graspable_cnt = torch.sum((target_labels > THRESH_BAD).long(),dim=2)
    graspable_label = (graspable_cnt>10) * objectness_label
    objectness_mask = (graspable_label > 0)
    center_loss = torch.sum((1-predict_view_score) * objectness_mask) / (objectness_mask.sum() + 1e-6)
    reg_loss = (1 - F.cosine_similarity(view_direction, target_direction, dim=-1)) * 0.5
    reg_loss = torch.sum(reg_loss * objectness_mask) / (objectness_mask.sum() + 1e-6)
    loss = reg_loss+0.1*center_loss
    end_points['loss/stage1_view_reg_loss'] = reg_loss
    end_points['loss/stage1_view_center_loss'] = center_loss
    return loss, end_points



def compute_grasp_loss_regression(end_points, episodic = False):
    top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    vp_rot = end_points['grasp_top_view_rot']  # (B, Ns, view_factor, 3, 3)
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_mask = torch.gather(objectness_label, 1, fp2_inds).bool()  # (B, Ns)
    # objectness_mask = end_points['graspable_mask']
    # process labels
    batch_grasp_label = end_points['batch_grasp_label']  # (B, Ns, A, D)
    batch_grasp_offset = end_points['batch_grasp_offset']  # (B, Ns, A, D, 3)
    batch_grasp_tolerance = end_points['batch_grasp_tolerance']  # (B, Ns, A, D)
    B, Ns, A, D = batch_grasp_label.size()

    # pick the one with the highest angle score
    top_view_grasp_angles = batch_grasp_offset[:, :, :, :, 0]  # (B, Ns, A, D)
    top_view_grasp_depths = batch_grasp_offset[:, :, :, :, 1]  # (B, Ns, A, D)
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :, 2]  # (B, Ns, A, D)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True)  # (B, Ns, 1, D)
    target_labels = torch.gather(batch_grasp_label, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_angles = torch.gather(top_view_grasp_angles, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_depths = torch.gather(top_view_grasp_depths, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    # end_points['target_angle'] = target_angles
    # end_points['target_width'] = target_widths
    # end_points['target_scores'] = target_labels
    # end_points['target_depths'] = target_depths
    target_tolerance = torch.gather(batch_grasp_tolerance, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)

    graspable_mask = (target_labels > THRESH_BAD)
    objectness_mask = objectness_mask.unsqueeze(-1).expand_as(graspable_mask)
    loss_mask = (objectness_mask & graspable_mask).float()

    end_points['label_mask'] = loss_mask
    full_mask = copy.deepcopy(loss_mask)
    if episodic:
        supervised_mask = end_points['sup_mask']
        supervised_mask = supervised_mask.unsqueeze(-1).expand_as(graspable_mask)

        loss_mask = loss_mask * supervised_mask

    # 1. grasp score loss
    depth_loss_mask = loss_mask.max(dim=2)[0].unsqueeze(-1).expand_as(loss_mask)
    target_labels_inds_ = target_labels_inds.transpose(1, 2)  # (B, 1, Ns, D)
    # grasp_score = torch.gather(end_points['grasp_score_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_score = end_points['grasp_score_pred']
    grasp_score_loss = huber_loss(grasp_score - target_labels, delta=1.0)
    grasp_score_loss = torch.sum(grasp_score_loss * depth_loss_mask) / (depth_loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_score_loss'] = grasp_score_loss

    # 2. inplane rotation regression loss
    target_angles_cls = target_labels_inds.squeeze(2)  # (B, Ns, D)
    # criterion_grasp_angle_class = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_sin2theta = end_points['grasp_angle_pred'][:,0]
    grasp_angle_cos2theta = end_points['grasp_angle_pred'][:, 1]
    # target_angles[target_angles>(0.5*np.pi)] -= np.pi
    target_sin2theta = torch.sin(target_angles * 2)
    target_cos2theta = torch.cos(target_angles * 2)
    grasp_angle_reg_loss = huber_loss(grasp_angle_sin2theta - target_sin2theta, delta=1.0) + huber_loss(grasp_angle_cos2theta - target_cos2theta, delta=1.0)
    grasp_angle_reg_loss = torch.sum(grasp_angle_reg_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    grasp_angle_pred = end_points['grasp_angle_value_pred']
    end_points['loss/stage2_grasp_angle_reg_loss'] = grasp_angle_reg_loss

    acc_mask_5 = ((torch.abs(grasp_angle_pred - target_angles) <= (5 / 180) * np.pi) | (
            torch.abs(grasp_angle_pred - target_angles) >= (175 / 180) * np.pi))
    end_points['stage2_grasp_angle_class_acc/5_degree'] = acc_mask_5[full_mask.bool()].float().mean()
    acc_mask_15 = ((torch.abs(grasp_angle_pred - target_angles) <= (15/180)*np.pi) | (
            torch.abs(grasp_angle_pred - target_angles) >= (165/180)*np.pi))
    end_points['stage2_grasp_angle_class_acc/15_degree'] = acc_mask_15[full_mask.bool()].float().mean()
    acc_mask_30 = ((torch.abs(grasp_angle_pred - target_angles) <= (30 / 180) * np.pi) | (
            torch.abs(grasp_angle_pred - target_angles) >= (150 / 180) * np.pi))
    end_points['stage2_grasp_angle_class_acc/30_degree'] = acc_mask_30[full_mask.bool()].float().mean()

    # 3. width reg loss
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_loss = huber_loss((grasp_width_pred - target_widths) / GRASP_MAX_WIDTH, delta=1)
    grasp_width_loss = torch.sum(grasp_width_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_width_loss'] = grasp_width_loss

    # 4. tolerance reg loss
    grasp_tolerance_pred = end_points['grasp_tolerance_pred']
    grasp_tolerance_loss = huber_loss((grasp_tolerance_pred - target_tolerance) / GRASP_MAX_TOLERANCE, delta=1)
    grasp_tolerance_loss = torch.sum(grasp_tolerance_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_tolerance_loss'] = grasp_tolerance_loss

    grasp_loss = grasp_score_loss + grasp_angle_reg_loss\
                 + grasp_width_loss + grasp_tolerance_loss

    return grasp_loss, end_points
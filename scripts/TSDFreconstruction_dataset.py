import open3d as o3d
import numpy as np
import os
from PIL import Image
from data_utils import CameraInfo,create_point_cloud_from_depth_image,transform_point_cloud


height = 720
width = 1280
root = "/data/graspnet"

for scene_id in range(0,190):
    scene_path = os.path.join(root,"scenes",'scene_{}'.format(str(scene_id).zfill(4)),"realsense")
    save_path = os.path.join(root, "fusion_scenes",'scene_{}'.format(str(scene_id).zfill(4)),"realsense")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    camera_poses = np.load(os.path.join(scene_path,"camera_poses.npy"))
    intrinsic_data = np.load(os.path.join(scene_path,"camK.npy"))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    to_world_mat = np.load(os.path.join(scene_path, "cam0_wrt_table.npy"))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1280,
        height=720,
        fx=intrinsic_data[0][0],
        fy=intrinsic_data[1][1],
        cx=intrinsic_data[0][2],
        cy=intrinsic_data[1][2]
    )
    camera = CameraInfo(1280.0, 720.0, intrinsic_data[0][0], intrinsic_data[1][1], intrinsic_data[0][2],
                        intrinsic_data[1][2], 1)
    for i in range(256):
        print("Integrate {:d}-th image into the volume.".format(i))
        color_path = os.path.join(scene_path, "rgb", str(i).zfill(4) + ".png")
        depth_path = os.path.join(scene_path, "depth", str(i).zfill(4) + ".png")
        seg_path = os.path.join(scene_path, "label", str(i).zfill(4) + ".png")

        seg = np.array(Image.open(seg_path))
        trans = np.dot(to_world_mat, camera_poses[i])
        rgb = np.array(Image.open(color_path))
        depth = np.array(Image.open(depth_path))
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(camera_poses[i]))

    pcd = volume.extract_point_cloud()
    # mesh.compute_vertex_normals()
    pcd.transform(to_world_mat)
    # pcd_remove, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
    #                                                     std_ratio=0.5)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    xyz = np.asarray(pcd.points)
    normal = np.asarray(pcd.normals)
    color = np.asarray(pcd.colors)
    save_dict = {
        'xyz':xyz,
        'color':color,
        'normal':normal
    }
    np.save(save_path+"/points.npy",save_dict)


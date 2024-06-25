import trimesh
import numpy as np
from mesh_to_sdf import mesh_to_sdf
import os

def grid_sample_sdf(mesh,save_dir):
    v = mesh.vertices
    x_max = np.max(v[:,0])
    x_min = np.min(v[:,0])
    y_max = np.max(v[:,1])
    y_min = np.min(v[:,1])
    z_max = np.max(v[:,2])
    z_min = np.min(v[:,2])

    padding = 0.03
    max_bound = max(x_max-x_min,y_max-y_min,z_max-z_min)
    step = (max_bound+0.06)/128
    # x_linespace = np.linspace(x_min-padding,x_max+padding,64)
    # y_linespace = np.linspace(y_min-padding,y_max+padding,64)
    # z_linespace = np.linspace(z_min-padding,z_max+padding,64)
    x_linespace = np.arange(x_min - padding, x_max + padding, step)
    y_linespace = np.arange(y_min - padding, y_max + padding, step)
    z_linespace = np.arange(z_min - padding, z_max + padding, step)
    x_grid,y_grid,z_grid = np.meshgrid(x_linespace,y_linespace,z_linespace)
    x_grid = np.expand_dims(np.transpose(x_grid,(1,0,2)),axis=3)
    y_grid = np.expand_dims(np.transpose(y_grid,(1,0,2)),axis=3)
    z_grid = np.expand_dims(np.transpose(z_grid,(1,0,2)),axis=3)
    grid_3d = np.concatenate([x_grid,y_grid,z_grid],axis=-1)
    print(grid_3d.shape)
    h,w,d,_ = grid_3d.shape
    grid_3d_ = grid_3d.reshape(-1,3)
    sdf = mesh_to_sdf(mesh,grid_3d_,surface_point_method='sample')
    sdf = sdf.reshape(h,w,d)
    save_path = os.path.join(save_dir,"grid_sampled_sdf.npz")
    np.savez(save_path,points=grid_3d,sdf=sdf)
    return

if __name__ == "__main__":
    import multiprocessing
    model_path = "/data/graspnet/models"
    obj_dirs = os.listdir(model_path)
    pool = multiprocessing.Pool(32)
    from tqdm import tqdm
    for i in tqdm(range(len(obj_dirs))):
        obj_dir = os.path.join(model_path, obj_dirs[i])
        print(obj_dir)
        obj_mesh = trimesh.load(os.path.join(obj_dir,"nontextured.ply"),process=False)
        pool.apply_async(grid_sample_sdf, args=(obj_mesh,obj_dir))
    pool.close()
    pool.join()
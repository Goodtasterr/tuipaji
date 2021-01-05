import open3d as o3d
import numpy as np
import os
import PIL.Image as Image
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
def pc_colors(arr):
    list = np.asarray([
        # [100, 100, 100],
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors)/255
def quat2rotation(Q):
    w, x, y, z = Q
    R = [[1-2*(y**2)-2*(z**2),2*x*y+2*w*z,2*x*z-2*w*y],
          [2*x*y-2*w*z,1-2*(x**2)-2*(z**2),2*y*z+2*w*x],
          [2*x*z+2*w*y,2*y*z-2*w*x,1-2*(x**2)-2*(y**2)]]
    # q0, q1, q2, q3 = Q
    # R = [[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    # [2 * (q1 * q2 + q0 * q3),q0**2 - q1 **2 + q2**2 - q3**2,2 * (q2 * q3 - q0 * q1)],
    # [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]]

    return np.asarray(R).astype(np.float32)

class tuipaji_pc():
    def __init__(self,pcd_root,quat,rotation_x=None):
        super().__init__()
        self.quat2rota = quat2rotation(quat)
        self.point_cloud_origin = np.asarray(o3d.io.read_point_cloud(pcd_root).points)
        self.point_cloud = np.dot(self.point_cloud_origin,self.quat2rota)
        trans = [[0, 1, 0],
                 [0, 0, 1],
                 [-1, 0, 0]]
        self.point_cloud = np.dot(self.point_cloud, trans)
        if rotation_x is not None:
            theta_x = rotation_x/180*np.pi
            rotation_matrix = [[1, 0, 0],
                          [0, np.cos(theta_x), -np.sin(theta_x)],
                          [0, np.sin(theta_x), np.cos(theta_x)]]
            self.point_cloud = np.dot(self.point_cloud,rotation_matrix)
        self.high = [np.min(self.point_cloud[:, 0]), np.max(self.point_cloud[:, 0])]
        print('down-->up:',self.high)
        self.width = [np.min(self.point_cloud[:, 1]), np.max(self.point_cloud[:, 1])]
        print('left-->right:',self.width)
        self.long = [np.min(self.point_cloud[:, 2]), np.max(self.point_cloud[:, 2])]
        print('front-->back:',self.long)

    def o3d_show(self,delta_h=None):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh.scale(10, center=mesh.get_center())
        if delta_h is not None:
            high_steps = []
            for i in range((int(self.high[1] / delta_h) - int(self.high[0] / delta_h)) + 3):
                high_steps.append(i * delta_h + delta_h * int(self.high[0] / delta_h - 1))
            print('contour  high:', high_steps)
            delta_x = 0.04
            indexs = np.zeros((self.point_cloud.shape[0])).astype(np.int)
            for j, high_step in enumerate(high_steps):
                index_step = (self.point_cloud[:, 0] > high_step - delta_x) & (
                            self.point_cloud[:, 0] < high_step + delta_x).astype(np.int)
                indexs += (index_step * (j % 3 + 1))

        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(self.point_cloud)
        if delta_h is not None:
            pcd_trans.colors = o3d.utility.Vector3dVector(pc_colors(indexs))
        o3d.visualization.draw_geometries([mesh,pcd_trans],window_name='window_name',
                                       width=1080, height=1080)

    def elevation_map(self,resolution = 0.1,show=''):
        width_steps = []
        for i in range((int(self.width[1] / resolution) - int(self.width[0] / resolution)) + 3):
            width_steps.append(i * resolution + resolution * int(self.width[0] / resolution - 1))
        print('contour width:', len(width_steps))
        long_steps = []
        for i in range((int(self.long[1] / resolution) - int(self.long[0] / resolution)) + 3):
            long_steps.append(i * resolution + resolution * int(self.long[0] / resolution - 1))
        print('contour  long:', len(long_steps))
        bias_width = abs(int(self.width[0] / resolution)) + 1
        bias_long = abs(int(self.long[0] / resolution)) + 1
        print('bias width & long:', bias_width, bias_long)

        pixel_points = (self.point_cloud[:, 1:] / resolution).astype(np.int) - 1
        pixel_points[:, 0] += bias_width
        pixel_points[:, 1] += bias_long
        bias_step = 5
        map_step = np.zeros([len(width_steps), len(long_steps)]) - bias_step
        for i in range(pixel_points.shape[0]):
            if (self.point_cloud[i][0] > map_step[pixel_points[i, 0], pixel_points[i, 1]]):
                map_step[pixel_points[i, 0], pixel_points[i, 1]] = self.point_cloud[i][0]
        xyh =[]
        for j in range((map_step.shape[0])):
            for k in range((map_step.shape[1])):
                xyh.append((j,k,map_step[j,k]))
        xyh_np = np.asarray(xyh).astype(np.float32)
        idx_empty = xyh_np<0.1
        xyh_false = xyh_np[idx_empty]
        xyh_true = xyh_np[(~idx_empty)]


        print(((map_step + bias_step) * 13).max(), ((map_step + bias_step) * 10).min())
        map_step2int = ((map_step + bias_step) * 13).astype(np.uint8)
        if show is 'map':
            map2img = Image.fromarray(map_step2int)
            map2img.show()
        elif show is '':
            pool_idx = map_step2int < 0.1
            map_pool = map_step2int + 0.0001
            map_show = map_step2int + map_pool * pool_idx
            print('map_show value:', ((map_pool).max(), (map_pool).min()))
            sns.set()
            ax = sns.heatmap(map_show, cmap='seismic', center=0, robust=True)
            plt.axis('equal')
            plt.show()
        return map_step

def tuipaji():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(10, center=mesh.get_center())

    # Q = np.asarray((-0.593, 0.113, 0.096, 0.792)).astype(np.float32) #last time quater
    Q = np.asarray((-0.091, -0.108,  -0.797, 0.588)).astype(np.float32)
    RR = quat2rotation(Q)

    print(RR,np.linalg.inv(RR)) #逆矩阵

    def colors(colors_arr,points_number):
        return np.repeat(colors_arr,points_number,axis=0)

    # pcd = o3d.io.read_point_cloud("./tuipaji_data/1607313674.393412828.pcd")
    pcd = o3d.io.read_point_cloud("./1222/1608617787805101.pcd")
    pcd_points = np.asarray(pcd.points)

    trans = [[0,1,0],
             [0,0,1],
             [-1,0,0]]
    theta_x = -20/180*np.pi
    rotation_x =[[1,0,0],
                [0,np.cos(theta_x),-np.sin(theta_x)],
                [0,np.sin(theta_x),np.cos(theta_x)]]
    pcd_points_trans = np.dot(pcd_points,(np.asarray(RR)))
    pcd_points_trans = np.dot(pcd_points_trans,np.asarray(trans))
    pcd_points_trans = np.dot(pcd_points_trans, np.asarray(rotation_x))

    points_number = pcd_points.shape[0]
    pcd.colors = o3d.utility.Vector3dVector(np.repeat([[0,1,1]],points_number,axis=0))

    #delta_h, delta_x
    delta_h = 1
    delta_x = 0.04
    resolution = 0.1
    high = [np.min(pcd_points_trans[:, 0]), np.max(pcd_points_trans[:, 0])]
    print('down-->up:',high)
    width = [np.min(pcd_points_trans[:, 1]), np.max(pcd_points_trans[:, 1])]
    print('left-->right:',width)
    long = [np.min(pcd_points_trans[:, 2]), np.max(pcd_points_trans[:, 2])]
    print('front-->back:',long)
    high_steps = []
    for i in range((int(high[1]/delta_h)-int(high[0]/delta_h))+3):
        high_steps.append(i*delta_h+delta_h*int(high[0]/delta_h-1))
    print('contour  high:',high_steps)
    width_steps = []
    for i in range((int(width[1]/resolution)-int(width[0]/resolution))+3):
        width_steps.append(i*resolution+resolution*int(width[0]/resolution-1))
    print('contour width:', len(width_steps))
    long_steps = []
    for i in range((int(long[1]/resolution)-int(long[0]/resolution))+3):
        long_steps.append(i*resolution+resolution*int(long[0]/resolution-1))
    print('contour  long:', len(long_steps))
    bias_width = abs(int(width[0]/resolution))+1
    bias_long = abs(int(long[0] / resolution)) +1
    print('bias width & long:',bias_width,bias_long)

    import time
    t0 = time.time()
    pixel_points = (pcd_points_trans[:,1:]/resolution).astype(np.int)-1
    pixel_points[:,0]+=bias_width
    pixel_points[:,1]+=bias_long
    bias_step = 5
    map_step = np.zeros([len(width_steps), len(long_steps)])-bias_step
    for i in range(pixel_points.shape[0]):
        if (pcd_points_trans[i][0]>map_step[pixel_points[i,0],pixel_points[i,1]]):
            map_step[pixel_points[i,0],pixel_points[i,1]]=pcd_points_trans[i][0]
    # print(map_step.min())
    print(time.time() - t0)
    print(((map_step+bias_step)*13).max(),((map_step+bias_step)*10).min())
    map_step2int = ((map_step+bias_step)*13).astype(np.uint8)
    kernel = np.ones((7, 7), np.float32) / 49
    pool_idx = map_step2int<0.1
    map_pool = map_step2int+0.0001
    # for i in range(4):
    #     map_pool = cv.filter2D(map_pool, -1, kernel)
    map_show = map_step2int + map_pool*pool_idx
    print('map_show value:',((map_pool).max(), (map_pool).min()))
    sns.set()
    ax = sns.heatmap(map_show,cmap='seismic',center=0,robust=True)
    plt.axis('equal')
    plt.show()

    print(type(map_step2int),map_step2int.shape)
    map2img = Image.fromarray(map_step2int)
    # map2img.show()


    indexs = np.zeros((pcd_points_trans.shape[0])).astype(np.int)
    for j, high_step in enumerate(high_steps):
        index_step = (pcd_points_trans[:,0]>high_step-delta_x)&(pcd_points_trans[:,0]<high_step+delta_x).astype(np.int)
        indexs+=(index_step*(j%3+1))

    # pcd_points_trans[:,0]= 0 #[高，宽，长]
    length = [np.min(pcd_points_trans[:,2]),np.max(pcd_points_trans[:,2])]
    width = [np.min(pcd_points_trans[:, 1]), np.max(pcd_points_trans[:, 1])]
    print(length,width)

    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pcd_points_trans)
    pcd_trans.colors = o3d.utility.Vector3dVector(pc_colors(indexs))

    # o3d.visualization.draw_geometries([mesh,pcd_trans],window_name='window_name',
    #                                    width=1080, height=1080)
if __name__ == '__main__':
    # tuipaji()

    Q = np.asarray((-0.091, -0.108, -0.797, 0.588)).astype(np.float32)
    test = tuipaji_pc(pcd_root="./1222/1608617787805101.pcd",quat=Q,rotation_x=-20)
    test.elevation_map(show='')

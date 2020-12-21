import open3d as o3d
import numpy as np
import os
import PIL.Image as Image


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

def tuipaji():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(10, center=mesh.get_center())

    w,x,y,z =   -0.593, 0.113, 0.096, 0.792
    RR = [[1-2*(y**2)-2*(z**2),2*x*y+2*w*z,2*x*z-2*w*y],
          [2*x*y-2*w*z,1-2*(x**2)-2*(z**2),2*y*z+2*w*x],
          [2*x*z+2*w*y,2*y*z-2*w*x,1-2*(x**2)-2*(y**2)]]


    # q0,q1,q2,q3 = -0.593, 0.113, 0.096, 0.792
    #
    #
    # RR = [[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    # [2 * (q1 * q2 + q0 * q3),q0**2 - q1 **2 + q2**2 - q3**2,2 * (q2 * q3 - q0 * q1)],
    # [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]]
    print(RR)

    def colors(colors_arr,points_number):

        return np.repeat(colors_arr,points_number,axis=0)

    pcd = o3d.io.read_point_cloud("./tuipaji_data/1607313674.393412828.pcd")
    pcd_points = np.asarray(pcd.points)
    pcd_points_trans = np.dot(pcd_points,np.asarray(RR))
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
    print('bias width long:',bias_width,bias_long)

    import time
    t0 = time.time()
    pixel_points = (pcd_points_trans[:,1:]/resolution).astype(np.int)-1
    pixel_points[:,0]+=bias_width
    pixel_points[:,1]+=bias_long
    map_step = np.zeros([len(width_steps), len(long_steps)])-3
    for i in range(pixel_points.shape[0]):
        if (pcd_points_trans[i][0]>map_step[pixel_points[i,0],pixel_points[i,1]]):
            map_step[pixel_points[i,0],pixel_points[i,1]]=pcd_points_trans[i][0]
    # print(map_step.min())
    print(time.time() - t0)
    map_step2int = ((map_step+3)*14).astype(np.uint8)

    print(type(map_step2int),map_step2int.shape)
    map2img = Image.fromarray(map_step2int)
    map2img.show()


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
    tuipaji()
    # im = Image.open('./map.pgm')
    #
    # im_arr = np.asarray(im)
    # im_arr_toimg = Image.fromarray(im_arr)
    # im_arr_toimg.show()
    # print(type(im_arr),im_arr.shape)


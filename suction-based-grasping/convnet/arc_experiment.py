import h5py
import cv2
import os
import socket
import copy

import numpy as np
import open3d as o3d

from camera import IntelCamera

def get_suction_point_3d(depth_image: np, suction_point: np, cam_instance):
    global pcd

    xyz = cam_instance.generate(depth_image)
    # pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
        
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    try:
        pcd.orient_normals_towards_camera_location()
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        suction_point = np.reshape(suction_point, (2, ))
    except:
        return 0
        
    ## creat xyz for the final suction point
    if cam_instance.device_product_line == "D400" or cam_instance.device_product_line == "AzureKinect":
        Z = depth_image[suction_point[0]][suction_point[1]]*0.001
    elif cam_instance.device_product_line == "L500":
        Z = depth_image[suction_point[0]][suction_point[1]]*0.00025
    X = (suction_point[1]-cam_instance.camera_mat[0][2])*Z/cam_instance.camera_mat[0][0]
    Y = (suction_point[0]-cam_instance.camera_mat[1][2])*Z/cam_instance.camera_mat[1][1]

    ## select most similar point among the generated points
    target = np.array([X, Y, Z])
    print("target:", target)
    target = np.reshape(target, (1, 3))
    index = np.linalg.norm((xyz-target), axis=1)
    index = np.argmin(index)
    point = np.asarray(pcd.points)[index]
    normal = np.asarray(pcd.normals)[index]

    return point, normal

def get_roi_mask(depth_image, prediction_map, cam_instance):
    '''
    - prediction map is 2d with single channel
    - could extract pixel index which got some level of probability
    - convert depth map to point cloud only for the selected pixel
    - filter points out of the roi
    - re-project points inside the roi
    - filter the pixel out of the roi using indices from depth map 
    '''

    width = cam_instance.intrinsic_o3d.width
    height = cam_instance.intrinsic_o3d.height
    fxy = cam_instance.intrinsic_o3d.get_focal_length()
    cxy = cam_instance.intrinsic_o3d.get_principal_point()

    max_pred = np.max(prediction_map)
    prediction_map = (prediction_map*(255/max_pred)).astype(np.uint8)
    prediction_mask = (prediction_map > 30).astype(int)

    # print(np.shape(depth_image))
    # print(np.shape(prediction_mask))

    filtered_depth = depth_image * prediction_mask
    ## render point cloud
    filtered_depth = filtered_depth.astype(np.uint16)
    cam_instance.generate(filtered_depth, downsample=False)
    xyz = cam_instance.crop_points()
    pcd = cam_instance.pcd
    # o3d.visualization.draw_geometries([pcd])

    roi_mask = np.full((height, width, 1), 0, dtype=np.uint8)
    for p in xyz:
        px = (fxy[0]*p[0] + cxy[0]*p[2])//p[2]
        py = (fxy[1]*p[1] + cxy[1]*p[2])//p[2]
        px = int(px)
        py = int(py)
        roi_mask[py][px] = [255]
    
    roi_mask = cv2.dilate(roi_mask, np.ones((2, 2), np.uint8))
    # cv2.imshow("roi_mask", roi_mask)
    roi_mask = roi_mask/255
    return roi_mask

cam = IntelCamera([])

pcd = o3d.geometry.PointCloud()

# set client information
SERVER_IP = '192.168.1.48'
SERVER_PORT = 9999
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

DATASock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

data_connection = DATASock.connect_ex(SERVER_ADDR)

if data_connection == 0:
    print("-"*100)
    print("CONNECTED TO SERVER!!")
    print("-"*100)

else:
    print("-"*100)
    print("**FAILED** TO CONNECT!!")
    print("-"*100)


for i in range(10):
    cam.stream()

rgb, depth = cam.stream()
cv2.imwrite("./color.png", rgb)
cv2.imwrite("./depth.png", depth)

os.system("th infer.lua")

file = h5py.File('results.h5')
result = file['results'][:]
prediction_map = result[0,1,:,:]
prediction_map = cv2.resize(prediction_map, (640, 480))

prediction_map[prediction_map >= 1] = 0.9999
prediction_map[prediction_map < 0] = 0

prediction_map = cv2.GaussianBlur(prediction_map, (25, 25), 5)

roi_mask = get_roi_mask(depth, prediction_map, cam)
prediction_map = prediction_map * roi_mask

max_prd = np.max(prediction_map)

## ind[1]: x, ind[0]: y
ind = np.unravel_index(np.argmax(prediction_map, axis=None), prediction_map.shape)

prediction_map = (prediction_map*(255/max_prd)).astype(np.uint8)
prediction_map = np.reshape(prediction_map, (480, 640, 1))
prediction_map = np.repeat(prediction_map, 3, -1)
heatmap = cv2.applyColorMap(prediction_map, cv2.COLORMAP_JET)

alpha = 0.6
combined_heatmap = cv2.addWeighted(heatmap, alpha, rgb, 1-alpha, 0)

cv2.circle(combined_heatmap, (ind[1], ind[0]), 3, (0, 255, 0), 2)
cv2.imshow("test", combined_heatmap)
cv2.waitKey()

suction_point_2d = np.array([int(ind[0]), int(ind[1])])
print("suction_point_2d:", suction_point_2d)

point, normal = get_suction_point_3d(depth, suction_point_2d, cam)
print("suction point:", point)

msg = str(point[0])+','+str(point[1])+','+str(point[2])+','+str(normal[0])+','+str(normal[1])+','+str(normal[2])
DATASock.send(msg.encode())
DATASock.close()
import os
import numpy as np
import sys
sys.path.append("..")
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from plyfile import PlyData, PlyElement


def save_point_cloud_to_ply(points, filename):
    """
    将点云保存为 PLY 文件
    :param points: 形状为 (N, 3) 的点云数据
    :param filename: 保存的文件名
    """
    # 创建一个包含点云数据的结构化数组
    vertices = np.array([(point[0], point[1], point[2]) for point in points],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # 创建 PlyElement 对象
    el = PlyElement.describe(vertices, 'vertex')
    # 写入 PLY 文件
    PlyData([el]).write(filename)
    



def visualize_point_cloud_with_axis(point_cloud,axis_vis=True):
    # 创建一个点云对象
    pcd = o3d.geometry.PointCloud()
    # 将numpy数组转换为open3d点云
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建坐标轴
    if axis_vis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # 创建一个窗口
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    vis.run()  # 运行可视化
    vis.destroy_window()  # 关闭窗口

    


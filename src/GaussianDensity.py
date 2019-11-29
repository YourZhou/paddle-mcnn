# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time : 2019-11-30

import math
import numpy as np



def fspecial(ksize, sigma):
    """
    Generates 2d Gaussian kernel
    :param ksize: an integer, represents the size of Gaussian kernel
    :param sigma: a float, represents standard variance of Gaussian kernel
    :return: 2d Gaussian kernel, the shape is [ksize, ksize]
    """
    # [left, right)
    left = -ksize / 2 + 0.5
    right = ksize / 2 + 0.5
    x, y = np.mgrid[left:right, left:right]
    # generate 2d Gaussian Kernel by normalization
    gaussian_kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2)).sum()
    sum = gaussian_kernel.sum()
    normalized_gaussian_kernel = gaussian_kernel / sum

    return normalized_gaussian_kernel


def get_avg_distance(position, points, k):
    """
    Computes the average distance between a pedestrian and its k nearest neighbors
    :param position: the position of the current point, the shape is [1,1]
    :param points: the set of all points, the shape is [num, 2]
    :param k: a integer, represents the number of mearest neibor we want
    :return: the average distance between a pedestrian and its k nearest neighbors
    """

    # in case that only itself or the k is lesser than or equal to num
    num = len(points)
    if num == 1:
        return 1.0
    elif num <= k:
        k = num - 1

    euclidean_distance = np.zeros((num, 1))
    for i in range(num):
        x = points[i, 1]
        y = points[i, 0]
        # Euclidean distance
        euclidean_distance[i, 0] = math.sqrt(math.pow(position[1] - x, 2) + math.pow(position[0] - y, 2))

    # the all distance between current point and other points
    euclidean_distance[:, 0] = np.sort(euclidean_distance[:, 0])
    avg_distance = euclidean_distance[1:k + 1, 0].sum() / k
    return avg_distance


# 使用动态高斯滤波器生成密度图(方法二，有bug)
def gaussian_filter_density(gt):
    # 使用高斯滤波变换生成密度图
    # Generates a density map using Gaussian filter transformation

    # 生成相应大小的零矩阵
    density = np.zeros(gt.shape, dtype=np.float32)

    # 返回所有非零数量（坐标）
    gt_count = np.count_nonzero(gt)

    # 如果全为零，说明场景没有人
    if gt_count == 0:
        # 直接返回孔密度图
        return density

    # 使用KDTree找出K个最近的邻居
    # FInd out the K nearest neighbours using a KDTree

    """
    1、读取ground-tround中非零坐标
    .ravel：扁平化数据
    zip()：打包为元组
    list():将元组转换为列表
    pts:标注点的列表
    """
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # 建立KDtree
    # leafsize:算法所处的点数
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # 查询 KDtree
    # query kdtree
    # distances距离
    # locations位置
    distances, locations = tree.query(pts, k=4)

    # enumerate() 函数用于将一个可遍历的数据对象
    # (如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标
    for i, pt in enumerate(pts):
        # 初始化数据
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        # 判断人数
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        # 高斯滤波器卷积
        # Convolve with the gaussian filter

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


# 使用高斯滤波器生成密度图（方法一）
def get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size, scaled_max_head_size):
    """
    Generates the correspoding ground truth density map
    :param scaled_crowd_img_size: the size of ground truth density map
    :param scaled_points: the position set of all points, but were divided into scale already
    :param knn_phase: True or False, determine wheather use geometry-adaptive Gaussian kernel or general one
    :param k: number of k nearest neighbors
    :param scaled_min_head_size: the scaled maximum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :param scaled_max_head_size:the scaled minimum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :return: density map, the shape is [scaled_img_size[0], scaled_img_size[1]]
    """

    h, w = scaled_crowd_img_size[0], scaled_crowd_img_size[1]

    density_map = np.zeros((h, w))
    # In case that there is no one in the image
    num = len(scaled_points)
    if num == 0:
        return density_map
    for i in range(num):
        # For a specific point in original points label of dataset, it represents as position[oy, ox],
        # so points[i, 1] is x, and points[i, 0] is y Also in case that the negative value
        x = min(h, max(0, abs(int(math.floor(scaled_points[i, 1])))))
        y = min(w, max(0, abs(int(math.floor(scaled_points[i, 0])))))
        # now for a specific point, it represents as position[x, y]

        position = [x, y]

        sigma = 1.5
        beta = 0.3
        ksize = 25
        if knn_phase:
            avg_distance = get_avg_distance(position, scaled_points, k=k)
            avg_distance = max(min(avg_distance, scaled_max_head_size), scaled_min_head_size)
            sigma = beta * avg_distance
            ksize = 1.0 * avg_distance

        # Edge processing
        x1 = x - int(math.floor(ksize / 2))
        y1 = y - int(math.floor(ksize / 2))
        x2 = x + int(math.ceil(ksize / 2))
        y2 = y + int(math.ceil(ksize / 2))

        if x1 < 0 or y1 < 0 or x2 > h or y2 > w:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(h, x2)
            y2 = min(w, y2)

            tmp = x2 - x1 if (x2 - x1) < (y2 - y1) else y2 - y1
            ksize = min(tmp, ksize)

        ksize = int(math.floor(ksize / 2))
        H = fspecial(ksize, sigma)
        density_map[x1:x1 + ksize, y1:y1 + ksize] = density_map[x1:x1 + ksize, y1:y1 + ksize] + H
    return np.asarray(density_map)

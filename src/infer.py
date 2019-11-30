#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time :2019-11-3.

import paddle.fluid as fluid
import numpy as np
import os
import random
import time
import utils
from data_util import *
from network import multi_column_cnn
from configs import *
import cv2 as cv

np.set_printoptions(threshold=np.inf)


# 密度图生成
def image_processing(input):
    # 高斯模糊
    kernel_size = (3, 3)
    sigma = 15
    r_img = cv.GaussianBlur(input, kernel_size, sigma)

    # 灰度图标准化
    norm_img = np.zeros(r_img.shape)
    norm_img = cv.normalize(r_img, norm_img, 0, 255, cv.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    # r_img = cv.resize(r_img, (720, 420))
    # utils.show_density_map(r_img)

    # 灰度图颜色反转
    imgInfo = norm_img.shape
    heigh = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((heigh, width, 1), np.uint8)
    for i in range(0, heigh):
        for j in range(0, width):
            grayPixel = norm_img[i, j]
            dst[i, j] = 255 - grayPixel

    # 生成热力图
    heat_img = cv.applyColorMap(dst, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    output = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

    return output


# 密度图与原图叠加
def image_add_heatmap(frame, heatmap, alpha=0.5):
    img_size = frame.shape
    heatmap = cv.resize(heatmap, (img_size[1], img_size[0]))
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
    cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图
    return frame


def load_image(img_path):
    # 读取初始图像原始
    ori_crowd_img = cv.imread(img_path)
    # 将图片俺按缩放倍数进行缩放
    scaled_img = cv.resize(ori_crowd_img, (256, 256))
    # scaled_img = cv.flip(scaled_img, 0)
    # 转换图像维度顺序

    im = scaled_img.transpose().astype('float32')
    im = np.expand_dims(im, axis=0)
    print(im.shape)

    # cropped_crowd_img = np.asarray(cropped_crowd_img)
    return im


def main():
    # 创建执行器
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 保存预测模型路径
    save_path = '../model/mcnn/infer_model5/'
    # 从模型中获取预测程序、输入数据名称列表、分类器
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

    img = load_image("./data_samples/IMG_1.jpg")
    # 执行预测
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: ((img - 127.5) / 128).astype(np.float32)},
                     fetch_list=target_var)

    # show_density_map(result[0, 0, :, :])

    num = np.sum(np.sum(result))
    print(num)
    result = result[0][0][-1]
    print(result.shape)
    show_density_map(result)
    result = result.transpose().astype('float32')

    show_density_map(result)
    ori_crowd_img = cv.imread("./data_samples/IMG_1.jpg")
    # ori_crowd_img = cv.flip(ori_crowd_img, 1)
    final_img = image_processing(result)
    final_img = image_add_heatmap(ori_crowd_img, final_img, 0.5)

    cv.putText(final_img, "P : " + str(int(num)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("really", final_img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time : 2019-09-28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid


def multi_column_cnn(inputs):
    mcnn = fluid.Scope()
    with fluid.scope_guard(mcnn):
        large = fluid.Scope()
        with fluid.scope_guard(large):
            large_column = fluid.layers.conv2d(input=inputs, num_filters=16, filter_size=9,
                                               name='conv1_1', padding='SAME')
            large_column = fluid.layers.conv2d(input=large_column, num_filters=32, filter_size=7,
                                               name='conv1_2', padding='SAME')
            large_column = fluid.layers.pool2d(input=large_column, pool_size=2, pool_stride=2, pool_type="max",
                                               name='pool1_1')
            large_column = fluid.layers.conv2d(input=large_column, num_filters=16, filter_size=7,
                                               name='conv1_3', padding='SAME')
            large_column = fluid.layers.pool2d(input=large_column, pool_size=2, pool_stride=2, pool_type="max",
                                               name='pool1_2')
            large_column = fluid.layers.conv2d(input=large_column, num_filters=8, filter_size=7,
                                               name='conv1_4', padding='SAME')

        medium = fluid.Scope()
        with fluid.scope_guard(medium):
            medium_column = fluid.layers.conv2d(input=inputs, num_filters=20, filter_size=7,
                                                name='conv2_1', padding='SAME')
            medium_column = fluid.layers.conv2d(input=medium_column, num_filters=40, filter_size=5,
                                                name='conv2_2', padding='SAME')
            medium_column = fluid.layers.pool2d(input=medium_column, pool_size=2, pool_stride=2, pool_type="max",
                                                name='pool2_1')
            medium_column = fluid.layers.conv2d(input=medium_column, num_filters=20, filter_size=5,
                                                name='conv2_3', padding='SAME')
            medium_column = fluid.layers.pool2d(input=medium_column, pool_size=2, pool_stride=2, pool_type="max",
                                                name='pool2_2')
            medium_column = fluid.layers.conv2d(input=medium_column, num_filters=10, filter_size=5,
                                                name='conv2_5', padding='SAME')

        small = fluid.Scope()
        with fluid.scope_guard(small):
            small_column = fluid.layers.conv2d(input=inputs, num_filters=24, filter_size=5,
                                               name='conv3_1', padding='SAME')
            small_column = fluid.layers.conv2d(input=small_column, num_filters=48, filter_size=3,
                                               name='conv3_2', padding='SAME')
            small_column = fluid.layers.pool2d(input=small_column, pool_size=2, pool_stride=2, pool_type="max",
                                               name='pool3_1')
            small_column = fluid.layers.conv2d(input=small_column, num_filters=24, filter_size=3,
                                               name='conv3_3', padding='SAME')
            small_column = fluid.layers.pool2d(input=small_column, pool_size=2, pool_stride=2, pool_type="max",
                                               name='pool3_2')
            small_column = fluid.layers.conv2d(input=small_column, num_filters=12, filter_size=3,
                                               name='conv3_4', padding='SAME')

        net = fluid.layers.concat([large_column, medium_column, small_column], axis=1)
        # print(net.shape)
        dmp = fluid.layers.conv2d(input=net, num_filters=1, filter_size=1, name='dmp_conv1', padding='SAME')

        return dmp


if __name__ == '__main__':
    input_img_data = fluid.data(name='image', shape=[-1, 3, 256, 256], dtype='float32')
    out = multi_column_cnn(input_img_data)
    print(out.shape)

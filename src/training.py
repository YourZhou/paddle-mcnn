# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time : 2019-11-30


import paddle.fluid as fluid
import os
from paddle.utils.plot import Ploter
import time
import matplotlib.pyplot as plt
import shutil
from data_util import *
from network import multi_column_cnn
from configs import *

np.set_printoptions(threshold=np.inf)


def train():
    dataset = 'A'
    # 训练数据集
    img_root_dir = r'D:/YourZhouProject/mcnn_project/pytorch_mcnn/part_' + dataset + r'_final/train_data/images/'
    gt_root_dir = r'D:/YourZhouProject/mcnn_project/pytorch_mcnn/part_' + dataset + r'_final/train_data/ground_truth/'
    # 测试数据集
    val_img_root_dir = r'D:/YourZhouProject/mcnn_project/pytorch_mcnn/part_' + dataset + r'_final/test_data/images/'
    val_gt_root_dir = r'D:/YourZhouProject/mcnn_project/pytorch_mcnn/part_' + dataset + r'_final/test_data/ground_truth/'

    # 训练数据集文件列表
    img_file_list = os.listdir(img_root_dir)
    gt_img_file_list = os.listdir(gt_root_dir)

    # 验证数据集文件列表
    val_img_file_list = os.listdir(val_img_root_dir)
    val_gt_file_list = os.listdir(val_gt_root_dir)

    # 获得训练参数信息
    cfig = ConfigFactory()

    # 变量定义
    input_img_data = fluid.data(name='input_img_data', shape=[-1, 3, 256, 256], dtype='float32')
    density_map_data = fluid.data(name='density_map_data', shape=[-1, 1, 64, 64], dtype='float32')

    # network generation网络生成
    inference_density_map = multi_column_cnn(input_img_data)

    # density map loss密度图损失
    # loss_sub = fluid.layers.elementwise_sub(density_map_data, inference_density_map)
    # density_map_loss = 0.5 * fluid.layers.reduce_sum(fluid.layers.square(loss_sub))
    squar = fluid.layers.square_error_cost(input=inference_density_map, label=density_map_data)
    cost = fluid.layers.sqrt(squar, name=None)
    # print(cost.shape)
    avg_cost = fluid.layers.mean(cost)
    # print(avg_cost.shape)

    # jointly training联合训练
    # 获取损失函数和准确率函数
    # joint_loss = density_map_loss
    # avg_cost = fluid.layers.mean(joint_loss)
    # acc = fluid.layers.accuracy(input=inference_density_map, label=density_map_data)

    # 获取训练和测试程序
    test_program = fluid.default_main_program().clone(for_test=True)

    # 我们使用的是Adam优化方法，同时指定学习率
    # 定义优化方法
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=cfig.lr)
    optimizer.minimize(avg_cost)

    # 定义一个使用GPU的执行器
    place = fluid.CUDAPlace(0)
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(fluid.default_startup_program())

    # 是否需要用feeder读入？
    # feeder = fluid.DataFeeder(place=place, feed_list=[input_img_data, density_map_data])

    # 获得训练日志保存地址
    file_path = cfig.log_router

    # 创建训练日志文件夹
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 创建模型训练日志文件
    if not os.path.exists(cfig.model_router):
        os.makedirs(cfig.model_router)
    log = open(cfig.log_router + cfig.name + r'_training.logs', mode='a+', encoding='utf-8')

    # 使用plt可视化训练
    train_prompt = "Train cost"
    cost_ploter = Ploter(train_prompt)

    def event_handler_plot(ploter_title, step, cost):
        cost_ploter.append(ploter_title, step, cost)
        cost_ploter.plot()

    # 开始训练
    step = 0

    for i in range(cfig.total_iters):
        # 训练
        for file_index in range(len(img_file_list)):
            # 获得图片路径
            img_path = img_root_dir + img_file_list[file_index]
            # 检查图片是否符合
            img_check = cv.imread(img_path)
            h, w, s = img_check.shape[0], img_check.shape[1], img_check.shape[2]
            if h < 256 or w < 256 or s < 3:
                continue

            # 获得标注文件路径
            gt_path = gt_root_dir + 'GT_' + img_file_list[file_index].split(r'.')[0]
            # 得到需要训练的图片、真实密度图、真实数量

            # 数据增强
            # Data_enhancement = np.random.randint(2)
            Data_enhancement = 1
            # print(Data_enhancement)
            if Data_enhancement == 0:
                img, gt_dmp, gt_count = read_crop_train_data(img_path, gt_path, scale=4)
            else:
                img, gt_dmp, gt_count = read_resize_train_data(img_path, gt_path, scale=4)

            # show_density_map(img[0, 0, :, :])
            # show_density_map(gt_dmp[0, 0, :, :])
            # plt.imshow(img[0, 0, :, :])
            # plt.show()
            # cv.imshow("123", img[0, :, :, :])
            # # cv.imshow("123", rea_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # train_img = ((img - 127.5) / 128).astype(np.float32)
            train_img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
            # 数据读入
            feed_dict = {'input_img_data': train_img,
                         'density_map_data': gt_dmp.astype(np.float32)}

            # 训练主题
            inf_dmp, loss = exe.run(program=fluid.default_main_program(),
                                    feed=feed_dict,
                                    fetch_list=[inference_density_map, avg_cost])

            # show_density_map(inf_dmp[0, 0, :, :])

            # if step % 100 == 0:
            #     event_handler_plot(train_prompt, step, loss[0])

            # 得到当前时间
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            # 训练过程查看
            format_str = 'iter={}, step={}, joint loss={}, inference={}, gt={} '
            log_line = format_time, img_file_list[file_index], format_str.format(i,
                                                                                 i * len(img_file_list) + file_index,
                                                                                 loss, inf_dmp.sum(), gt_count)
            log.writelines(str(log_line) + '\n')
            print(log_line)

            step += 1

        # 测试
        if i % 5 == 0:
            # 训练的过程中可以保存预测模型，用于之后的预测。
            # 保存预测模型
            save_path = cfig.model_router + 'infer_model' + str(i) + '/'
            # 创建保持模型文件目录
            os.makedirs(save_path)
            # 保存预测模型
            fluid.io.save_inference_model(save_path,
                                          feeded_var_names=[input_img_data.name],
                                          target_vars=[inference_density_map],
                                          executor=exe)

            val_log = open(cfig.log_router + cfig.name + r'_validating_' + str(i) + '_.logs', mode='w',
                           encoding='utf-8')
            absolute_error = 0.0
            square_error = 0.0
            # validating验证
            for file_index in range(len(val_img_file_list)):

                # 获得测试图片路径
                img_path = val_img_root_dir + val_img_file_list[file_index]
                # 检查图片是否符合
                img_check = cv.imread(img_path)
                h, w, s = img_check.shape[0], img_check.shape[1], img_check.shape[2]
                if h < 256 or w < 256 or s < 3:
                    continue

                # 获得测试标注文件路径
                gt_path = val_gt_root_dir + 'GT_' + val_img_file_list[file_index].split(r'.')[0]
                # 得到需要测试的图片、真实密度图、真实数量
                img, gt_dmp, gt_count = read_resize_train_data(img_path, gt_path, scale=4)

                # 数据读入
                feed_dict = {'input_img_data': ((img - 127.5) / 128).astype(np.float32),
                             'density_map_data': gt_dmp.astype(np.float32)}

                # 训练主题
                inf_dmp, loss = exe.run(program=test_program,
                                        feed=feed_dict,
                                        fetch_list=[inference_density_map, avg_cost])

                # 测试过程查看
                format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                format_str = 'Test iter={}, step={}, joint loss={}, inference={}, gt={} '
                absolute_error = absolute_error + np.abs(np.subtract(gt_count, inf_dmp.sum())).mean()
                square_error = square_error + np.power(np.subtract(gt_count, inf_dmp.sum()), 2).mean()
                log_line = format_time, val_img_file_list[file_index], format_str.format(i,
                                                                                         file_index, loss,
                                                                                         inf_dmp.sum(), gt_count)
                val_log.writelines(str(log_line) + '\n')
                print(log_line)

            # 获得测试结果（均方误差和均方根误差）
            mae = absolute_error / len(val_img_file_list)
            rmse = np.sqrt(absolute_error / len(val_img_file_list))
            val_log.writelines(str('MAE_' + str(mae) + '_MSE_' + str(rmse)) + '\n')
            val_log.close()
            print(str('MAE_' + str(mae) + '_MSE_' + str(rmse)))


if __name__ == '__main__':
    train()

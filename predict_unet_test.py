
import glob
import numpy as np
import torch
import os
import cv2
from unet import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1,bilinear=False)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    checkpoint = torch.load('model_pth',map_location=device)
    net.load_state_dict(checkpoint['net'])
    # 测试模式
    net.eval()
    # 读取所有图片路径
    Test_Data_path = 'D:\\BaiduNetdiskDownload\\data\\data\\test\\'
    tests_path = glob.glob(Test_Data_path + '*.png')
    # 遍历所有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图单通道
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # pytorch要求的格式（batch_size,c,w,h）
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)
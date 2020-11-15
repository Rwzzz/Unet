# #清除 预测文件
import os

path = 'D:\BaiduNetdiskDownload\data\data\\test\\'

dirs = os.listdir(path)
for file in dirs:
    print(file)
    if file.find('res') != -1:
        os.remove(path+file)

# import os
# import cv2
# import numpy as np
# path = 'D:\Rwz\labelme\car_license\\01-90_88-343&484_529&548-529&547_350&548_355&491_534&490-0_0_14_32_30_25_25-86-34.jpg'
# path_res =  'D:\Rwz\labelme\car_license\\01-90_88-343&484_529&548-529&547_350&548_355&491_534&490-0_0_14_32_30_25_25-86-34_res.png'
# raw = cv2.imread(path)
# image_res = cv2.imread(path_res)
# imgs = np.hstack([raw, image_res])
# # 展示多个
# cv2.imshow("picture", imgs)
# # 等待关闭
# cv2.waitKey(0)

#one-hot编码
# import torch as t
# import numpy as np
#
# batch_size = 8
# class_num = 10
# label = np.random.randint(0,class_num,size=(batch_size,1))
# label = t.LongTensor(label)
# print(label)
# y_one_hot = t.zeros(batch_size,class_num).scatter_(1,label,1)
# print(y_one_hot)
import torch
import numpy as np
'''
gt = np.random.randint(0,5, size=[15,15])  #先生成一个15*15的label，值在5以内，意思是5类分割任务
gt = torch.LongTensor(gt)

print(gt)

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


gt_one_hot = get_one_hot(gt, 5)
print(gt_one_hot)
print(gt_one_hot.shape)

print(gt_one_hot.argmax(-1) == gt)  # 判断one hot 转换方式是否正确，全是1就是正确的
'''



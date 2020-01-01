import torch
import numpy as np

out = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(out.shape)
fusion_weight = torch.ones(3, 8, 8)
inter = 8 // 3
weight = out
for i in range(3):
    for j in range(3):
        w = weight[i][j]
        print(i, j, w)
        fusion_weight[:, i * inter + 1:(i + 1) * inter + 1, j * inter + 1:(j + 1) * inter + 1] = w
print(fusion_weight.shape, fusion_weight)

videos1 = torch.from_numpy(np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]))
videos2 = torch.from_numpy(np.array([[15, 25, 35], [45, 55, 65], [75, 85, 95]]))
video_new = torch.mul(videos1,out) + torch.mul(videos2,(1-out))
print(video_new)

distance = torch.from_numpy(np.array([[ 1080.6152,   603.4716,   880.7473,   651.7961,   934.9957]]))
probability = torch.nn.functional.softmax(-distance)
loss = - torch.log(probability)
print (probability,loss)

distance2 = torch.from_numpy(np.array([[ 10.6152,   6.4716,   8.7473,   6.7961,   9.9957]]))
probability2= torch.nn.functional.softmax(-distance2)
loss2 = - torch.log(probability2)
print (probability2,loss2)


#normolize big loss
distance3 = distance/(torch.mean(distance,dim=1))
probability3 = torch.nn.functional.softmax(-distance3)
loss3 = - torch.log(probability3)
print(probability3,loss3)
# tensor([[ 0.1446,  0.2569,  0.1839,  0.2423,  0.1723]], dtype=torch.float64) tensor([[ 1.9339,  1.3593,  1.6932,  1.4175,  1.7585]], dtype=torch.float64)

# normolize small loss
distance4= distance2/(torch.mean(distance2,dim=1))
probability4 = torch.nn.functional.softmax(-distance4)
loss4 = - torch.log(probability4)
print(probability4,loss4)
# tensor([[ 0.1536,  0.2497,  0.1912,  0.2404,  0.1652]], dtype=torch.float64) tensor([[ 1.8735,  1.3875,  1.6544,  1.4256,  1.8009]], dtype=torch.float64)


from sklearn.metrics.pairwise import cosine_similarity

def cosine_distance(matrix1, matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance

matrix1=np.array([[1,1],[1,2]])
matrix2=np.array([[2,1],[2,2],[2,3]])
cosine_dis=cosine_distance(matrix1,matrix2)
print ('cosine_dis:',cosine_dis)

cosine_dis2 = cosine_similarity(matrix1,matrix2)
print('cosine_dis2:',cosine_dis2)

import subprocess



print('-------------------------------------------------------------------------------------------------')
# input distance numpy [8,640]   ---> [640,8 ]  -----> conv1d[0.6,1,0.6]  ------> [640,8]  -----> [8,640]
dist = np.random.random((8,640*8))
print (dist.shape)

dist = np.transpose(dist,(1,0))
print(dist.shape)


import torch.nn as nn
from  torch.autograd import Variable
import torch.nn.functional as F

class TemporalLayer(nn.Module):
    def __init__(self):
        super(TemporalLayer,self).__init__()
        # kernal = [[0,0,0],[0.6,1,0.6],[0,0,0]]
        kernal = [0.6,1,0.6]
        kernal = torch.FloatTensor(kernal)
        print('kernal:', kernal.shape)
        kernal = kernal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        print(kernal)
        self.weight = nn.Parameter(data=kernal,requires_grad=False)

    def forward(self, x):
        x = F.conv1d(x,self.weight,padding=(0,1))
        return x


x = Variable(torch.randn(1,1,640*8,8))
myTemporalLayer = TemporalLayer()
out = myTemporalLayer(x)
print(out.shape)

# verify accuracy
video = torch.FloatTensor(np.array([[10.0, 20, 30], [40, 50, 60], [70, 80, 90],[700, 800, 900]]))
video = Variable(video)
video = video.unsqueeze(0).unsqueeze(0)
output = myTemporalLayer(video)
output = output.view(output.shape[2],output.shape[3])
print('video:',video,'\n','output:',output)


for i in range(5):
    print(i)

for i in range(3,5):
    print(i)


print('-------------------------------------------')
seg_len = 2
probe_seg_nums=150

def get_gallery_pool_ids_random(probe_seg_nums):
    gallery_seg_nums = 64 * 10 // seg_len
    aim_galllery_pool_ids=[]
    for i in range(probe_seg_nums):
        aim_id = np.random.randint(gallery_seg_nums)
        aim_galllery_pool_ids.append(aim_id)
    aim_galllery_pool_ids = np.array(aim_galllery_pool_ids)
    return aim_galllery_pool_ids

result = get_gallery_pool_ids_random(probe_seg_nums)
print (result)

print('--------------------------------------------')

x = np.random.random((3,4))

mu = 0
sigma = 0.12
x_out = x + np.random.normal(mu,sigma)

print(x,x_out,x_out-x)

import copy

x_out=copy.deepcopy(x)
for i in range(x.shape[0]):
  for j in range(x.shape[1]):
     x_out[i][j] = x_out[i][j] +np.random.normal(mu,sigma)
print(x,x_out,x_out-x)


x_out=copy.deepcopy(x)
x_out = x_out +np.random.normal(mu,sigma,size=x_out.shape)
print(x,x_out,x_out-x)


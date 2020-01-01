import cv2
import os
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from utils import *
from epoch_dataloader import VideoDataset
from models import model_resnet18, model_resnet50

class TrainNetwork():
    def __init__(self, loss_path, ckp_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size=10, resnet_model='resnet50', num_classes = num_classes_train):
        # get params
        self.loss_path = loss_path
        self.ckp_path = ckp_path
        self.epoch_nums = epoch_nums
        self.batch_size = batch_size
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.lr_step_size = lr_step_size
        self.resnet_model = resnet_model
        self.num_classes = num_classes
       
        # mkdir ckg_path
        if not os.path.exists(self.ckp_path):
            os.makedirs(self.ckp_path)

        # load model
        if (self.resnet_model == 'resnet18'):
           self.mymodel = model_resnet18(num_classes=self.num_classes)
        elif (self.resnet_model == 'resnet50'):
           self.mymodel = model_resnet50(num_classes=self.num_classes)
        self.mymodel.train()
        self.mymodel.cuda()
        print('model loaded.')

        # define video_dataloader
        self.myDataset = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train')
        self.myDataloader = DataLoader(self.myDataset, batch_size= self.batch_size, shuffle=True, num_workers=8)


    def finetune_model(self,data_aug ='None',pre_model= None):
        '''
        :param data_aug = 'None' use trainData
        :param data_aug = 'aug_seg_T' use trainData2.3
        '''
        file = open(self.loss_path,'w')

        if (pre_model):
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model,'loaded.')

        # define dataloader
        if (data_aug == 'None'):
            dataset_train = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size= self.batch_size, shuffle=True, num_workers=8)

        elif(data_aug == 'aug_seg_T'):
            dataset_train = VideoDataset(TRAIN_LIST, TrainAugSegDatasets_DIR_2_3, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        else:
            print('data aug error.')
            return 0
        # define some params
        optimizer_1 = optim.SGD(self.mymodel.convnet.parameters(), lr=self.lr_1, momentum=0.9)
        optimizer_2 = optim.SGD(self.mymodel.fc.parameters(), lr=self.lr_2, momentum=0.9)
        scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=self.lr_step_size, gamma=0.1)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=self.lr_step_size, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # training
        for epoch in range(self.epoch_nums):
            scheduler_1.step()
            scheduler_2.step()
            running_loss=0.0
            for i_batch, sample_batched in enumerate(dataloader_train):
                #get the inputs
                video,label=sample_batched['video'],sample_batched['label']
                video_shape = video.shape
                video = video.view(-1,video_shape[2],video_shape[3],video_shape[4])
                label= label.view((label.shape[0])).type(torch.LongTensor)
                video,label=Variable(video).cuda(),Variable(label.cuda())

                # zero the parameter gradients
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()

                # forward
                feature, output=self.mymodel(video)
                feature = feature.view(video_shape[0],video_shape[1],-1)

                if(data_aug == 'aug_frame_gaussian'):
                     feature = feature.cpu().detach().numpy()  #[b_s,frame_num,2048]
                     mu,sigma = 0,0.3
                     size=(feature.shape[0],seg_len,feature.shape[2])
                     feature[:,:seg_len,:] = feature[:,:seg_len,:]+ np.random.normal(mu,sigma,size=size)
                     feature = torch.FloatTensor(feature).cuda()
                     # print('aug_frame_gaussian ok.')

                feature = torch.mean(feature,dim=1)

                output = self.mymodel.fc(feature)
                loss=criterion(output,label)
                loss.backward()
                optimizer_1.step()
                optimizer_2.step()

                #caculate accuracy
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                predicted_y = np.argmax(output, axis=1)
                accuracy = np.mean(label == predicted_y)

                # print statistics
                running_loss=running_loss+loss.data[0]
                if i_batch % 50 == 49:
                    print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i_batch + 1, running_loss / 50, accuracy))
                    print ('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i_batch + 1, running_loss / 50, accuracy),file=file)
                    running_loss = 0.0
            save_model_path= self.ckp_path +'model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.state_dict(),save_model_path)




if __name__ == '__main__':

    # fintune model on source data
    '''
    loss_path, ckp_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, resnet_model = './result/train_exp_baseline2.txt', './result/exp_baseline2/',6, 6, 0.0001,0.001,10,'resnet50'
    myTrainNetwork = TrainNetwork(loss_path,ckp_path,epoch_nums,batch_size,lr_1, lr_2, lr_step_size,resnet_model)
    myTrainNetwork.finetune_model(data_aug='None', pre_model=None)
    '''
   
    # you need to make sure that you have generate the augmented data 

    # further finetune model on augmented source data
    loss_path, ckp_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, resnet_model = './result/train_exp_baseline2_step2.txt', './result/exp_baseline2_step2/',6, 6, 0.00001,0.0001,6,'resnet50'
    myTrainNetwork = TrainNetwork(loss_path,ckp_path,epoch_nums,batch_size,lr_1, lr_2, lr_step_size,resnet_model)
    #myTrainNetwork.finetune_model(pre_model='./result/exp_baseline2/model6.pkl')
    myTrainNetwork.finetune_model(data_aug='aug_seg_T',pre_model='./result/exp_baseline2/model6.pkl')

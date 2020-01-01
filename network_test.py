import cv2
import os
import  numpy as np
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
from classifier import Classifier
from models import model_resnet18, model_resnet50,TemporalLayer
from episode_novel_dataloader import EpisodeDataloader
from generate_gallery_videos import generate_gallery_videos

class TestNetwork():
    def __init__(self, test_result_txt,resnet_model='resnet50', classifier='protonet',L2 = True,num_classes=num_classes_train,mode='test'):
       # set params
        self.test_result_txt =  test_result_txt
        self.resnet_model = resnet_model
        self.classifier = classifier
        self.L2 = L2
        self.num_classes = num_classes
        self.mode = mode

       # load model
        if (self.resnet_model == 'resnet18'):
           self.mymodel = model_resnet18(num_classes=self.num_classes)
        elif (self.resnet_model == 'resnet50'):
           self.mymodel = model_resnet50(num_classes=self.num_classes)
        self.mymodel.eval()
        self.mymodel.cuda()
        print('model loaded.')

        # define episode_dataloader
        self.myEpisodeDataloader = EpisodeDataloader(mode=self.mode)

        # define one-shot classifier
        self.myClassifier = Classifier(classifier = self.classifier)


    def generate_epoch_features(self,videos,L2=False,support_x_frames=None):
        video_features=[]
        for i in range(videos.shape[0]):  # (D,h,w,c)
            video =videos[i]
            # print(video.shape)
            if(support_x_frames):
                video = video[0:support_x_frames[i]]

            ## print(video.shape)
            input = Variable(video).cuda()
            feature,output = self.mymodel(input)
            #feature = self.generate_epoch_features_2(video,L2)
            # print(feature.shape)
            if (L2):
               feature= torch.nn.functional.normalize(feature, p=2, dim=1)
            feature = feature.cpu().detach().numpy()
            feature = np.mean(feature, axis=0)
            video_features.append(feature)
        video_features = np.array(video_features)
        return video_features

    def generate_epoch_features_2(self,videos,L2=False):
        # print(videos.shape)
        video_features=[]
        batch_size = 96
        inter = videos.shape[0]//batch_size
        for i in range(inter):
            tmp_videos = videos[i*batch_size:(i+1)*batch_size]
            tmp_videos = Variable(tmp_videos).cuda()
            tmp_features, outputs= self.mymodel(tmp_videos)
            if (L2):
                tmp_features = torch.nn.functional.normalize(tmp_features, p=2, dim=1)
            tmp_videos = tmp_videos.cpu().detach().numpy()
            tmp_features = tmp_features.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            # print(i * batch_size, (i + 1) * batch_size, 'video:',tmp_videos.shape,'feature:',tmp_features.shape)
            video_features.append(tmp_features)

        if ( inter * batch_size  < videos.shape[0]):
            tmp_videos = videos[inter*batch_size: ]
            tmp_videos = Variable(tmp_videos).cuda()
            tmp_features,outputs = self.mymodel(tmp_videos)
            if (L2):
                tmp_features = torch.nn.functional.normalize(tmp_features, p=2, dim=1)
            tmp_videos = tmp_videos.cpu().detach().numpy()
            tmp_features = tmp_features.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            # print( inter * batch_size ,videos.shape[0], 'video:', tmp_videos.shape, 'feature:', tmp_features.shape)
            video_features.append(tmp_features)
        video_features = np.concatenate(video_features,axis=0)
        return video_features


    # AUGMENT TRAIN DATASETS 3(ALL_FRAMES) (semantic closest + temporal consistence)
    def temporal_convolution_flating_layer(self,distance):
        # define Temporal Model
        myTemporalLayer = TemporalLayer()
        myTemporalLayer.cuda()
        # prepare input data
        distance = np.transpose(distance,(1,0))         # [8,640*8]   --->  [640*8,8]
        distance = torch.FloatTensor(distance)
        distance = Variable(distance).cuda()
        distance = distance.unsqueeze(0).unsqueeze(0)   # [1,1,640*8,8]
        # forward
        distance_new = myTemporalLayer(distance)        # [1,1,640*8,8]
        distance_new = distance_new.view(distance_new.shape[2], distance_new.shape[3])  #[640*8, 8]
        distance_new = distance_new.cpu().detach().numpy()
        distance_new = np.transpose(distance_new,(1,0)) # [8,640*8]
        return distance_new

    def video_segment_augmentation(self,video_probe_seg,seg_id,gallery_seg,data_aug=None):
        aug_video = copy.deepcopy(video_probe_seg)
        if(data_aug=='aug_image_gaussian'):
            aug_video[seg_id] = aug_video[seg_id] + np.random.normal(0,0.3,(seg_len,3,224,224))
        elif(data_aug =='aug_frame_gaussian'):
            pass
        else:
            aug_video[seg_id] = gallery_seg
        aug_video = np.resize(aug_video,(VIDEO_FRAMES,3,224,224))
        # print(aug_video.shape)
        return aug_video


    def test_network_baseline(self,pre_model=None):
        if(pre_model):
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model,'loaded.')
        self.mymodel.eval()

        # init file
        self.acc_file = open(self.test_result_txt, "w")

        epoch_nums = EPISODE_NUMS[self.mode]
        accs = []
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader.get_episode()
            support_x, support_y, query_x, query_y,support_x_frames = data['support_x'], data['support_y'], data['query_x'], data['query_y'],data['support_x_frames']
            # print(support_x.shape,support_y.shape,query_x.shape,query_y.shape,support_x_frames)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()

            # get support_x features and query_x features
            support_features = self.generate_epoch_features(support_x,self.L2,support_x_frames)
            query_features = self.generate_epoch_features(query_x,self.L2)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = support_features, support_y, query_features, query_y

            # one-shot classifier
            predicted_y = self.myClassifier.predict(data_result)
            acc = np.mean(query_y == predicted_y)

            # show result
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs))
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs),file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)


    def test_network_aug_segment(self,pre_model = None, data_aug = 'aug_seg_T'):
        '''
        :param data_aug  'aug_seg_T:' semantic + temporal consistence augment
        '''
        if pre_model:
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model, 'loaded.')
        self.mymodel.eval()

        # init file
        self.acc_file = open(self.test_result_txt, "w")

        print("preaparing gallery segments.")
        #get gallery_videos and gallery_videos
        gallery_videos = generate_gallery_videos()                         # [640,16,3,224,224]
        gallery_segments = gallery_videos.view(-1, seg_len, 3, 224, 224)   # [640*8,2,3,224,224]
        gallery_videos = gallery_videos.view(-1, 3, 224, 224)              # [640*16,3,224,224]
        gallery_features = self.generate_epoch_features_2(gallery_videos, self.L2)  # [640*16,2048]
        gallery_seg_features = np.resize(gallery_features,(640 * VIDEO_FRAMES // seg_len, seg_len, 2048))  # [640*8,2,2048]
        gallery_seg_features = np.mean(gallery_seg_features, axis=1)       # [640*8,2048]
        # print('g_video:',gallery_videos.shape,'g_seg:',gallery_segments.shape,'g_fea:',gallery_seg_features.shape)

        epoch_nums = EPISODE_NUMS[self.mode]
        accs = []
        num_segs = VIDEO_FRAMES // seg_len
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader.get_episode()
            support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
            query_features = self.generate_epoch_features(query_x, self.L2)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()
            support_segments = support_x.view(-1, seg_len, 3, 224, 224)   # [n_way*k_shot,16,3,224,224]   --> [n_way*k_shot*8,2,3,224,224]
            support_videos = support_x.view(-1, 3, 224, 224)              # [n_way*k_shot*16,3,224,224]
            support_features = self.generate_epoch_features_2(support_videos, self.L2)  # [n_way*k_shot*16,2048]
            support_seg_features = np.resize(support_features, (n_way*k_shot*VIDEO_FRAMES // seg_len, seg_len, 2048))  # [n_way*k_shot*8,2,2048]
            support_seg_features = np.mean(support_seg_features, axis=1)  # [n_way*k_shot*8,2048]

            if (data_aug =='aug_seg_T'):
                distance = cdist(support_seg_features, gallery_seg_features, 'euclidean')
                distance = self.temporal_convolution_flating_layer(distance)
                # choose aim gallery_seg for per support_seg
                gallery_pool_ids = np.argsort(distance, axis=1)
                gallery_pool_ids = gallery_pool_ids[:, :1]
                # print('gallery_pool_ids:',gallery_pool_ids)
                gallery_pool_ids = np.resize(gallery_pool_ids, (n_way * k_shot, num_segs))
            else:
                print('data_aug error.')
                return 0

      # generate aug_video_features,aug_video_labels
            aug_video_features = []
            aug_video_labels = []
            support_segments = support_segments.view(n_way * k_shot, num_segs, seg_len, 3, 224, 224)
            # support_seg_features = np.resize(support_seg_features,(n_way*k_shot,num_segs,))
            for i in range(gallery_pool_ids.shape[0]):
                video_label = support_y[i]
                video_probe_seg = support_segments[i]
                # video_probe_seg_feature = support_seg_features[i]
                # video_probe_feature = np.mean(video_probe_seg_feature,axis=0)
                video_probe_feature = support_seg_features[i]
                # print('label:',video_label,'probe_seg:',video_probe_seg.shape,'probe_fea:',video_probe_feature.shape)
                aug_video_features.append(video_probe_feature)
                aug_video_labels.append(video_label)
                for seg_id in range(num_segs):
                    gallery_seg_id = gallery_pool_ids[i][seg_id]
                    gallery_seg = gallery_segments[gallery_seg_id]
                    # print(seg_id,gallery_seg.shape)
                    video_aug = self.video_segment_augmentation(video_probe_seg, seg_id, gallery_seg,data_aug)
                    # get video_aug feature
                    video_aug = torch.FloatTensor(video_aug)
                    video_aug = Variable(video_aug).cuda()
                    video_aug_feature, _ = self.mymodel(video_aug)
                    if (self.L2):
                        video_aug_feature = torch.nn.functional.normalize(video_aug_feature, p=2, dim=1)
                    video_aug_feature = video_aug_feature.cpu().detach().numpy()
                    video_aug_feature = np.mean(video_aug_feature, axis=0)
                    # print('video_aug:',video_aug.shape,'video_aug_feature:',video_aug_feature.shape,'video_label:',video_label)
                    aug_video_features.append(video_aug_feature)
                    aug_video_labels.append(video_label)
            aug_video_features = np.array(aug_video_features)
            aug_video_labels = np.array(aug_video_labels)
            # print('aug_video_features:',aug_video_features.shape,'aug_video_labels:',aug_video_labels.shape)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = aug_video_features, aug_video_labels, query_features, query_y
            # print(data_result['support_feature'].shape, data_result['support_y'].shape,data_result['query_feature'].shape, data_result['query_y'].shape)

            # one-shot classifier
            predicted_y = self.myClassifier.predict(data_result)
            acc = np.mean(query_y == predicted_y)

            # show result
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs))
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs),file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)






if __name__ == '__main__':
    # test baseline
    acc_path = './result/acc_exp_baseline2_2000_5shot.txt'
    myTestNetwork = TestNetwork(acc_path,'resnet50','protonet',True)
    myTestNetwork.test_network_baseline(pre_model='./result/exp_baseline2/model6.pkl')
    myTestNetwork.test_network_aug_segment(pre_model='./result/exp_baseline2/model6.pkl')

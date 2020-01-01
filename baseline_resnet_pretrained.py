import cv2
import os
import  numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from utils import *
from episode_novel_dataloader import EpisodeDataloader
from network_test import generate_gallery_list,generate_gallery_videos
from network_test import generate_gallery_videos_2

def get_whole_video_from_video_info(video_info, mode='train',frame_dir=KINETICS_FRAME_DIR):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :return: torch.Size([all_video_frames, 3, 242, 242])
    '''
    video_frame_path = os.path.join(frame_dir, video_info)
    all_frame_count = len(os.listdir(video_frame_path)) - 1

    myTransform = transforms(mode=mode)
    video = []
    frames_path = []
    for i in range(all_frame_count):
        image_id = i + 1
        s = "%05d" % image_id
        image_name = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_name)
        frames_path.append(image_path)
        image = Image.open(image_path)
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image = myTransform(image)
        video.append(image)
    video = torch.stack(video, 0)
    # print(video.shape, frames_path)
    return video, np.array(frames_path)

def generate_prototypes_tensor_lowerdim(data):
    '''
    data:  dict{'support_feature'[],'support_y'[],'query_feature'[],'query_y'[]}
    return: prototype_ids, prototype_features
    '''
    support_feature, support_y, query_x, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']

    # get prototype ids and prototype_features
    prototype_ids = []
    prototype_features = []

    dict = {}
    for i in range(support_y.shape[0]):
        classId = support_y[i]
        # turn feature from [video_frames,dim] to [dim]
        video_feature = support_feature[i]
        # video_feature = np.mean(video_feature, axis=0)
        if classId not in dict.keys():
            dict[classId] = [video_feature]
        else:
            dict[classId].append(video_feature)
    # print(dict.keys())

    for classId in dict.keys():
        prototype_ids.append(classId)
        prototype_feature = np.array(dict[classId])
        prototype_feature = np.mean(prototype_feature, axis=0)
        prototype_features.append(prototype_feature)
    prototype_features = np.array(prototype_features)
    # print(prototype_ids,prototype_features.shape)
    # [0.0, 1.0, 2.0, 3.0, 4.0] (5, 2048)
    return (prototype_ids, prototype_features)

def one_shot_classifier_prototype_lowerdim(data):
    '''
    data: dict{'support_feature[],'support_y'[],'query_feature'[],'query_y'[]}
    return : loss ,accuracy
    '''
    # get input
    support_feature, support_y, query_feature, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']

    # get prototypes_ids and prototype_features
    prototype_ids, prototype_features = generate_prototypes_tensor_lowerdim(data)
    # print(prototype_ids,prototype_features.shape)

    # get distance
    query_features = []
    for i in range(query_y.shape[0]):
        query_feature = query_feature[i]
        # query_feature = np.mean(query_feature, axis=0)
        query_features.append(query_feature)
    query_features = np.array(query_features)

    distance = cdist(query_features, prototype_features,metric='euclidean')

    # get probability
    distance = torch.FloatTensor(distance)
    probability = torch.nn.functional.softmax(-distance)
    # print('probability:',probability,'query_y:',query_y)

    # get loss
    loss = 0
    for i in range(query_y.shape[0]):
        label = query_y[i]
        classid = prototype_ids.index(label)
        loss = loss + (- torch.log(probability[i][classid]))
    loss = loss / query_y.shape[0]
    # print('prototype loss:',loss)
    # loss.backward()

    # caculate accuracy
    label = query_y
    probability = probability.data.cpu().numpy()
    predicted_y = np.argmax(probability, axis=1)
    accuracy = np.mean(label == predicted_y)
    # print('P: ', 'label:', label, 'predicted_y:', predicted_y, 'accuracy:', accuracy, 'loss:', loss)

    return loss, accuracy

class model_resnet18(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet18,self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        # self.fc = nn.Linear(2048,num_classes)
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        output = self.fc(feature)
        return feature,output

class model_resnet50(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet50,self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        # self.fc = nn.Linear(8192,num_classes)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        output = self.fc(feature)
        return feature,output

# dataloader for trainAug
class TrainAugSegDataset(Dataset):
    def __init__(self, info_txt = TrainAugSegDatasetsInfo_LIST, root_dir = TrainAugSegDatasets_DIR):
        # set params
        self.info_txt = info_txt
        self.root_dir = root_dir

        # read info_list
        self.info_list = open(self.info_txt).readlines()

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        video_info_splits = info_line.strip('\n').split(',')
        video_name ,video_label =video_info_splits[0].strip(' '),video_info_splits[1].strip(' ')
        video_path = self.root_dir + video_name
        video = np.load(video_path)     # [16,3,224,224]


        # verfiy video
        '''
        videoA = torch.FloatTensor(copy.deepcopy(video))
        for k in range(videoA.shape[0]):
            image = videoA[k]
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            # use sigmod to [0,1]
            image = 1.0 / (1 + np.exp(-1 * image))
            # transfer to [0,255]
            image = np.round(image * 255)
            image_path = './result/probe_image' + str(k) + '.jpg'
            cv2.imwrite(image_path, image)
        '''

        sample = {'video': video, 'label': [int(video_label)]}
        sample['video'] = torch.FloatTensor(sample['video'])
        sample['label'] = torch.FloatTensor(sample['label'])
        return sample

# for temporal convolution flating layer
class TemporalLayer(nn.Module):
    def __init__(self):
        super(TemporalLayer,self).__init__()
        # kernal = [[0,0,0],[0.6,1,0.6],[0,0,0]]
        # kernal = [0.6, 1, 0.6]
        kernal = [lamda1,lamda2,lamda1]
        kernal = torch.FloatTensor(kernal)
        # kernal = kernal.unsqueeze(0).unsqueeze(0)
        kernal = kernal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernal,requires_grad=False)

    def forward(self, x):
        # x = F.conv1d(x,self.weight,padding=1)
        x = F.conv1d(x, self.weight, padding=(0,1))
        return x



class TestBaselineNetwork():
    def __init__(self, test_result_txt,resnet_model='resnet18',classifier='protonet',L2 = True,num_classes=num_classes_train):
       # set params
        self.test_result_txt =  test_result_txt
        self.resnet_model = resnet_model
        self.classifier = classifier
        self.L2 = L2
        self.num_classes = num_classes

       # load model
        if (self.resnet_model == 'resnet18'):
           self.mymodel = model_resnet18(num_classes=self.num_classes)
        elif (self.resnet_model == 'resnet50'):
           self.mymodel = model_resnet50(num_classes=self.num_classes)
        self.mymodel.eval()
        self.mymodel.cuda()
        print('model loaded.')

        # define episode_dataloader
        self.myEpisodeDataloader_test = EpisodeDataloader(mode='test')

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

    def test_network_baseline(self,pre_model=None,data_aug = None):
        if(pre_model):
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model,'loaded.')
        self.mymodel.eval()

        # init file
        self.acc_file = open(self.test_result_txt, "w")

        epoch_nums = test_episodes
        accs = []
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader_test.get_episode()
           
            #  support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
       
            support_x, support_y, query_x, query_y,support_x_frames = data['support_x'], data['support_y'], data['query_x'], data['query_y'],data['support_x_frames']
            # print(support_x.shape,support_y.shape,query_x.shape,query_y.shape,support_x_frames)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()

            # get support_x features and query_x features
            support_features = self.generate_epoch_features(support_x,self.L2,support_x_frames)
            query_features = self.generate_epoch_features(query_x,self.L2)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = support_features, support_y, query_features, query_y
            # print(data_result['support_feature'].shape,data_result['support_y'].shape,data_result['query_feature'].shape,data_result['query_y'].shape)

            # use aug_video_features and aug_video_labels to learn one-shot classifier
            # 1. protoNet
            # 2. SVM
            # 3. KNN
            # 4. logistic regression
            # 5. classifier NN
            if (self.classifier == 'protonet'):
                loss, acc = one_shot_classifier_prototype_lowerdim(data_result)

            elif (self.classifier == 'SVM'):
                classifier_SVM = SVC(C=10)   # TODO CHOOSE C AND OTHER PARAMS
                classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
                predicted_y = classifier_SVM.predict(data_result['query_feature'])
                # print('query_y:', query_y, 'pre_y:', predicted_y)
                acc = np.mean(query_y == predicted_y)
                loss = 0
            elif (self.classifier == 'KNN'):
                classifier_KNN = KNeighborsClassifier(n_neighbors= k_shot)
                classifier_KNN.fit(data_result['support_feature'],data_result['support_y'])
                predicted_y = classifier_KNN.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0
            elif(self.classifier == 'cosine'):
                from sklearn.metrics.pairwise import cosine_similarity
                distance_cosine = cosine_similarity(data_result['query_feature'],data_result['support_feature'])
                predicted_y = np.argsort(-distance_cosine)
                predicted_y = predicted_y[:,0]
                acc = np.mean(query_y == predicted_y)
                loss =0
            else:
                print('classifier type error.')

            # test query ,get loss and acc
            print('epoch:', epoch,                'acc:', acc,'avg_acc:',np.mean(accs))
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc,'avg_acc:',np.mean(accs),file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)

    #  TEST_VIDEO_AUGMENTATION
    def video_block_augmentation(self,video_base, video_gallery, block_m = block_m_test):
        # verify video_probe and video_gallery  TODO
        '''
        for k in range(video_base.shape[0]):
            image = video_base[k]
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            # use sigmod to [0,1]
            image = 1.0 / (1 + np.exp(-1 * image))
            # transfer to [0,255]
            image = np.round(image * 255)
            image_path = './result/probe_image' + str(k) + '.jpg'
            cv2.imwrite(image_path, image)

        for k in range(video_gallery.shape[0]):
            image = video_gallery[k]
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            # use sigmod to [0,1]
            image = 1.0 / (1 + np.exp(-1 * image))
            # transfer to [0,255]
            image = np.round(image * 255)
            image_path = './result/gallery_image' + str(k) + '.jpg'
            cv2.imwrite(image_path, image)
        '''

        # random choose m blocks (m<=block_m)
        m = random.randint(1,block_m)
        block_arrays = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        aim_block_arrays= random.sample(block_arrays,m)

        # generate video_aug
        video_aug = video_base.clone()
        inter = 224//3
        for aim_block in aim_block_arrays:
            i,j=aim_block[0],aim_block[1]
            video_aug[:,:, i*inter+1:(i+1)*inter+1, j*inter+1:(j+1)*inter+1] = video_gallery[:,:, i*inter+1:(i+1)*inter+1, j*inter + 1:(j+1)*inter+1]

        '''
        for k in range(video_aug.shape[0]):
            image = video_aug[k]
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            # use sigmod to [0,1]
            image = 1.0 / (1 + np.exp(-1 * image))
            # transfer to [0,255]
            image = np.round(image * 255)
            image_path = './result/aug_image' + str(k) + '.jpg'
            cv2.imwrite(image_path, image)
        '''

        return video_aug

    def test_network_aug_block(self):
        self.mymodel.eval()
        # init file
        self.acc_file = open(self.test_result_txt, "w")

        # get gallery_videos and gallery_videos
        gallery_videos = generate_gallery_videos()
        gallery_features = self.generate_epoch_features(gallery_videos,self.L2)

        epoch_nums = test_episodes
        accs = []
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader_test.get_episode()
            support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
            support_features = self.generate_epoch_features(support_x,self.L2)
            query_features = self.generate_epoch_features(query_x,self.L2)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()

            # print('suppport_videos:',support_x.shape,'gallery_videos:',gallery_videos.shape)
            # print('support_features:',support_features.shape,'gallery_features:',gallery_features.shape)

            # get distance
            distance = cdist(support_features,gallery_features,metric='euclidean')
            # print('distance:',distance.shape)

            # choose aim gallery videos for per support
            gallery_pool_ids = np.argsort(distance, axis=1)
            gallery_pool_ids = gallery_pool_ids[:, :gallery_num_per_class_test]
            # print('gallery_pool_ids:',gallery_pool_ids)


            # generate aug_video_features,aug_video_labels
            aug_video_features=[]
            aug_video_labels=[]
            for i in range(gallery_pool_ids.shape[0]):
                video_label = support_y[i]
                video_probe = support_x[i]
                video_feature_probe = support_features[i]
                aug_video_features.append(video_feature_probe)
                aug_video_labels.append(video_label)

                # get video_probe and video_gallery ==> video_aug
                for gallery_id in gallery_pool_ids[i]:
                    video_gallery = gallery_videos[gallery_id]
                    video_gallery = torch.FloatTensor(video_gallery)
                    video_aug = self.video_block_augmentation(video_base=video_probe,video_gallery=video_gallery)
                    video_aug = Variable(video_aug).cuda()
                    video_aug_feature, output = self.mymodel(video_aug)
                    if(self.L2):
                        video_aug_feature = torch.nn.functional.normalize(video_aug_feature, p=2, dim=1)
                    video_aug_feature = video_aug_feature.cpu().detach().numpy()
                    video_aug_feature = np.mean(video_aug_feature, axis=0)
                    aug_video_features.append(video_aug_feature)
                    aug_video_labels.append(video_label)
            aug_video_features = np.array(aug_video_features)
            aug_video_labels = np.array(aug_video_labels)
            # print('aug_video_features:',aug_video_features.shape,'aug_video_labels:',aug_video_labels.shape)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = aug_video_features, aug_video_labels, query_features, query_y
            # print(data_result['support_feature'].shape, data_result['support_y'].shape,data_result['query_feature'].shape, data_result['query_y'].shape)

            # get query_feature
            # query_features=[]
            # for video_query in query_x:
            #     video_query = torch.FloatTensor(video_query)
            #     video_query = Variable(video_query).cuda()
            #     video_query_feature, output = self.mymodel(video_query)
            #     # add L2 norm
            #     video_query_feature = torch.nn.functional.normalize(video_query_feature, p=2, dim=1)
            #     video_query_feature = video_query_feature.cpu().detach().numpy()
            #     query_features.append(video_query_feature)
            # query_features = np.array(query_features)
            # print('query_video_features:',query_features.shape,'query_labels:',query_y.shape)

            # use aug_video_features and aug_video_labels to learn one-shot classifier
            # 1. protoNet
            # 2. SVM
            # 3. KNN
            # 4. logistic regression
            # 5. classifier NN
            if (self.classifier == 'protonet'):
                loss, acc = one_shot_classifier_prototype_lowerdim(data_result)

            elif (self.classifier == 'SVM'):
                classifier_SVM = SVC(C=10)     # TODO CHOOSE C AND OTHER PARAMS
                classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
                predicted_y = classifier_SVM.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0
            elif (self.classifier == 'KNN'):
                classifier_KNN = KNeighborsClassifier(n_neighbors= k_shot)
                classifier_KNN.fit(data_result['support_feature'],data_result['support_y'])
                predicted_y = classifier_KNN.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0

            else:
                print('classifier type error.')


            # test query ,get loss and acc
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc)
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc, file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)

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

    def test_network_aug_segment(self,pre_model = None):
        if pre_model:
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model,'loaded.')
            
        self.mymodel.eval()

        # init file
        self.acc_file = open(self.test_result_txt, "w")

        # get gallery_videos and gallery_videos
        gallery_videos = generate_gallery_videos()
        gallery_segments = gallery_videos.view(-1,seg_len,3,224,224)
        gallery_seg_features = self.generate_epoch_features(gallery_segments,self.L2)
        # print('g_video:',gallery_videos.shape,'g_seg:',gallery_segments.shape,'g_fea:',gallery_seg_features.shape)

        epoch_nums = test_episodes
        accs = []
        num_segs = VIDEO_FRAMES // seg_len
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader_test.get_episode()
            support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
            query_features = self.generate_epoch_features(query_x, self.L2)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()
            support_segments = support_x.view(-1,seg_len,3,224,224)
            support_seg_features = self.generate_epoch_features(support_segments,self.L2)
            # print('support_x:',support_x.shape,'s_seg:',support_segments.shape,'s_fea:',support_seg_features.shape)
            # get seg_distance
            distance = cdist(support_seg_features,gallery_seg_features,'euclidean')
            # choose aim gallery_seg for per support_seg
            gallery_pool_ids = np.argsort(distance, axis=1)
            gallery_pool_ids = gallery_pool_ids[:, :1]
            # print('gallery_pool_ids:',gallery_pool_ids)
            gallery_pool_ids = np.resize(gallery_pool_ids,(n_way*k_shot,num_segs))


            # generate aug_video_features,aug_video_labels
            aug_video_features = []
            aug_video_labels = []
            support_segments = support_segments.view(n_way*k_shot,num_segs,seg_len,3,224,224)
            # support_seg_features = np.resize(support_seg_features,(n_way*k_shot,num_segs,))
            for i in range(gallery_pool_ids.shape[0]):
                video_label = support_y[i]
                video_probe_seg= support_segments[i]
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
                    video_aug = self.video_segment_augmentation(video_probe_seg,seg_id,gallery_seg)
                    # get video_aug feature
                    video_aug = torch.FloatTensor(video_aug)
                    video_aug = Variable(video_aug).cuda()
                    video_aug_feature,_ = self.mymodel(video_aug)
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

            # get query_feature
            # query_features=[]
            # for video_query in query_x:
            #     video_query = torch.FloatTensor(video_query)
            #     video_query = Variable(video_query).cuda()
            #     video_query_feature, output = self.mymodel(video_query)
            #     # add L2 norm
            #     video_query_feature = torch.nn.functional.normalize(video_query_feature, p=2, dim=1)
            #     video_query_feature = video_query_feature.cpu().detach().numpy()
            #     query_features.append(video_query_feature)
            # query_features = np.array(query_features)
            # print('query_video_features:',query_features.shape,'query_labels:',query_y.shape)

            # use aug_video_features and aug_video_labels to learn one-shot classifier
            # 1. protoNet
            # 2. SVM
            # 3. KNN
            # 4. logistic regression
            # 5. classifier NN
            if (self.classifier == 'protonet'):
                loss, acc = one_shot_classifier_prototype_lowerdim(data_result)

            elif (self.classifier == 'SVM'):
                classifier_SVM = SVC(C=10)     # TODO CHOOSE C AND OTHER PARAMS
                classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
                predicted_y = classifier_SVM.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0
            elif (self.classifier == 'KNN'):
                classifier_KNN = KNeighborsClassifier(n_neighbors= k_shot)
                classifier_KNN.fit(data_result['support_feature'],data_result['support_y'])
                predicted_y = classifier_KNN.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0

            else:
                print('classifier type error.')


            # test query ,get loss and acc
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc)
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc, file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)


    # AUGMENT TRAIN DATASETS AND FINETUNE_RESNET_MODEL
    def generate_trainAug_datasets(self,train_info=TRAIN_LIST, trainAug_info=TrainAugSegDatasetsInfo_LIST,trainAug_dir=TrainAugSegDatasets_DIR):
        train_video_list = open(train_info).readlines()
        file = open(trainAug_info, 'w')

        # get gallery_videos and gallery_seg_videos and gallery_seg_features
        gallery_videos = generate_gallery_videos()                        # tensor [640,16,3,224,224]
        gallery_segments = gallery_videos.view(-1, seg_len, 3, 224, 224)  # tensor [640*8,2,3,224,224]
        gallery_seg_features = self.generate_epoch_features(gallery_segments, self.L2)  # numpy [640*8,2048]

        for i in range(len(train_video_list)):
            base_video_info = train_video_list[i].strip('\n')
            base_video = get_video_from_video_info(base_video_info,mode='test')       # tensor [16,3,224,224]
            base_label = get_label_from_video_info(base_video_info)
            # save base label
            base_video_name = 'video' + str(i) + '_' + str(0) + '.npy'
            base_video_path = trainAug_dir + base_video_name
            print('base_video:', base_video.shape, 'base_label:', base_label, base_video_name)
            print(base_video_name,',', base_label, file=file)
            np.save(base_video_path, base_video.numpy())

            # aug videos for base_video
            base_segments = base_video.view(-1, seg_len, 3, 224, 224)           # tensor [8,2,3,224,224]
            base_seg_features = self.generate_epoch_features(base_segments, self.L2)  # numpy [8,2048]

            distance = cdist(base_seg_features, gallery_seg_features, 'euclidean')
            # print('distance:',distance.shape)                # [8, 640*8]

            # choose aim gallery_seg for per support_seg
            gallery_pool_ids = np.argsort(distance, axis=1)  # [8,640*8]
            gallery_pool_ids = gallery_pool_ids[:, :1]       # [8,1]
            # print('gallery_pool_ids:',gallery_pool_ids)

            # get video_aug
            for j in range(gallery_pool_ids.shape[0]):
                gallery_seg_id = gallery_pool_ids[j]
                gallery_seg = gallery_segments[gallery_seg_id]    # tensor [2,3,224,224]
                aug_video = base_segments.clone()                 # tensor [8,2,3,224,224]
                aug_video[j] = gallery_seg
                aug_video = aug_video.view(base_video.shape)      # tensor [16,3,224,224]
                # save aug video
                aug_video_name = 'video' + str(i) + '_' + str(1+j) + '.npy'
                aug_video_path = trainAug_dir + aug_video_name
                print('aug_video:', aug_video.shape, 'aug_label:', base_label, aug_video_name)
                print(aug_video_name,',', base_label, file=file)
                np.save(aug_video_path, aug_video.numpy())

    def finetune_model(self,loss_path,model_path,epoch_nums,batch_size,lr_1,lr_2,lr_step_size = 10,data_aug ='None',pre_model= None):
        self.mymodel.train()
        file = open(loss_path,'w')

        if (pre_model):
            self.mymodel.load_state_dict(torch.load(pre_model))

        # define dataloader
        if (data_aug == 'None'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg'):
            dataset_train = TrainAugSegDataset()
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        else:
            pass

        # define some params
        optimizer_1 = optim.SGD(self.mymodel.convnet.parameters(), lr=lr_1, momentum=0.9)
        optimizer_2 = optim.SGD(self.mymodel.fc.parameters(), lr=lr_2, momentum=0.9)
        scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=lr_step_size, gamma=0.1)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=lr_step_size, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # training
        for epoch in range(epoch_nums):
            scheduler_1.step()
            scheduler_2.step()
            running_loss=0.0
            for i_batch, sample_batched in enumerate(dataloader_train):
                #get the inputs
                video,label=sample_batched['video'],sample_batched['label']
                if (data_aug=='aug_seg'):
                    video = video.view(-1,16,3,224,224)
                    label = label.view(-1)
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
            save_model_path= model_path +'model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.state_dict(),save_model_path)


    # AUGMENT TRAIN DATASETS 2 (ALL_FRAMES)(semantic closest) AND FINETUNE_RESNET_MODEL
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

    def generate_trainAug_datasets_2_1(self,train_info = TRAIN_LIST,trainAug_dir = TrainAugSegDatasets_DIR_2_1):
        train_video_list = open(train_info).readlines()

        # get gallery_videos and gallery_seg_videos and gallery_seg_features
        gallery_videos,gallery_videos_frames_dir = generate_gallery_videos_2()  # tensor [640,16,3,224,224]  # array[640,16]
        gallery_videos_frames_dir = np.resize(gallery_videos_frames_dir,(640*16))       # [640*16]

        # gallery_segments = gallery_videos.view(-1, seg_len, 3, 224, 224)  # tensor [640*8,2,3,224,224]
        # gallery_seg_features = self.generate_epoch_features(gallery_segments, self.L2)  # numpy [640*8,2048]
        gallery_videos = gallery_videos.view(-1,3,224,224)       # []
        gallery_features = self.generate_epoch_features_2(gallery_videos,self.L2)    # [640*16,2048]
        gallery_seg_features = np.resize(gallery_features,(640*VIDEO_FRAMES//seg_len,seg_len,2048))  #[640*8,2,2048]
        gallery_seg_features = np.mean(gallery_seg_features,axis=1)
        # print(gallery_seg_features.shape)     # [640*8,2048]

        # print('gallery_videos:', gallery_videos.shape, 'gallery_videos_frames_dir', gallery_videos_frames_dir.shape)

        for  i in range(len(train_video_list)):
            train_info = train_video_list[i].strip('\n')
            train_video,train_frames_dir = get_whole_video_from_video_info(train_info)
            # print(train_video.shape, train_frames_dir.shape,train_frames_dir[:2])    # tensor[300,3,224,224]  [300]

            norm_frames = train_video.shape[0]//seg_len * seg_len
            # print('video_frames:',train_video.shape[0],'norm_frames:',norm_frames)  # example 301, seq_len=2, norm =300

            train_video,train_frames_dir = train_video[:norm_frames],train_frames_dir[:norm_frames]
            # print('train_video:',train_video.shape,'train_frames_dir:',train_frames_dir.shape)

            # train_segments = train_video.view(-1,seg_len,3,224,224)     # tensor [150,2,3,224,224]
            # train_seg_features = self.generate_epoch_features(train_segments)  # [150,2048]

            train_features = self.generate_epoch_features_2(train_video, self.L2)  # [300,2048]
            train_seg_features = np.resize(train_features,(norm_frames// seg_len, seg_len, 2048))  # [150,2,2048]
            train_seg_features = np.mean(train_seg_features, axis=1)   # [150,2048]
            # print(train_seg_features.shape)  # [150,2048]


            distance = cdist(train_seg_features, gallery_seg_features, 'euclidean')
            # print('distance:',distance.shape)                # [150 640*8]

            # choose aim gallery_seg for per support_seg
            gallery_pool_ids = np.argsort(distance, axis=1)  # [150,640*8]
            gallery_pool_ids = gallery_pool_ids[:, :1]       # [150,1]


            # get aug_video_frames and save them in new dir
            for i in range(norm_frames):
                # change the frames
                if  i % VIDEO_FRAMES ==0 :
                    gallery_seg_id = gallery_pool_ids[i // seg_len]
                    for j in range(seg_len):
                        # '/DATACENTER/2/lovelyqian/Kinetics/Kinetics/miniKinetics_frames/blowing glass/pkakR3JSTuU_000006_000016/image_00122.jpg'
                        init_frame_path = train_frames_dir[i+j]
                        gallery_frame_path = gallery_videos_frames_dir[gallery_seg_id * 2+j][0]
                        # print(init_frame_path)
                        # print(gallery_frame_path)

                        init_frame_path_splits = init_frame_path.split('/')
                        # print(init_frame_path_splits)

                        class_name = init_frame_path_splits[-3]
                        video_name = init_frame_path_splits[-2]
                        image_name = init_frame_path_splits[-1]
                        dst_class_path = os.path.join(trainAug_dir, class_name)
                        dst_directory_path = os.path.join(dst_class_path, video_name)
                        # if not os.path.exists(dst_class_path):
                        #     os.mkdir(dst_class_path)
                        # if not os.path.exists(dst_directory_path):
                        #     os.mkdir(dst_directory_path)
                        cmd = 'cp "%s" "%s/%s"'%(gallery_frame_path, dst_directory_path,image_name)
                        print(i,cmd)
                        subprocess.call(cmd, shell=True)
                elif  i%VIDEO_FRAMES<seg_len:
                    pass
                else:
                    pass
                    # init_frame_path = train_frames_dir[i]
                    # # print(init_frame_path)
                    #
                    # init_frame_path_splits = init_frame_path.split('/')
                    # # print(init_frame_path_splits)
                    #
                    # class_name = init_frame_path_splits[-3]
                    # video_name = init_frame_path_splits[-2]
                    # image_name = init_frame_path_splits[-1]
                    # dst_class_path = os.path.join(trainAug_dir, class_name)
                    # dst_directory_path = os.path.join(dst_class_path, video_name)
                    # if not os.path.exists(dst_class_path):
                    #     os.mkdir(dst_class_path)
                    # if not os.path.exists(dst_directory_path):
                    #     os.mkdir(dst_directory_path)
                    # cmd = 'cp "%s" "%s"'%(init_frame_path, dst_directory_path)
                    # subprocess.call(cmd, shell=True)
                    # print(cmd)

    def finetune_model_2(self,loss_path,model_path,epoch_nums,batch_size,lr_1,lr_2,lr_step_size = 10,data_aug ='None',pre_model= None):
        '''
        :param data_aug = 'None' use trainData
        :param data_aug = 'aug_seg' use trainData2.1
        :param data_aug = 'aug_seg_T_0.6' use trainData2.2
        :param data_aug = 'aug_seg_T' use trainData2.3
        :param data_aug = 'aug_seg_random' use trainData2.4
        :param data_aug = ‘aug_seg_T_seg_len_1' use trainData2.5
        :param data_aug = ‘aug_seg_T_seg_len_4' use trainData2.6
        :param data_aug = ‘aug_image_gaussian' use trainData and all gaussian in image
        :param data_aug = ‘aug_frame_gaussian' use trainData and all gaussian in frame
        '''
        self.mymodel.train()
        file = open(loss_path,'w')

        if (pre_model):
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model,'loaded.')

        # define dataloader
        if (data_aug == 'None'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST,TrainAugSegDatasets_DIR_2_1 , mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg_T_0.6'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST,TrainAugSegDatasets_DIR_2_2 , mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg_T'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, TrainAugSegDatasets_DIR_2_3, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg_random'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, TrainAugSegDatasets_DIR_2_4, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg_T_seg_len_1'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, TrainAugSegDatasets_DIR_2_5, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_seg_T_seg_len_4'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, TrainAugSegDatasets_DIR_2_6, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_image_gaussian'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train',data_aug=data_aug)
            dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True, num_workers=8)
        elif(data_aug == 'aug_frame_gaussian'):
            from epoch_dataloader import VideoDataset
            dataset_train = VideoDataset(TRAIN_LIST, KINETICS_FRAME_DIR, mode='train')
            dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True, num_workers=8)

        else:
            print('data aug error.')
            return 0
        # define some params
        optimizer_1 = optim.SGD(self.mymodel.convnet.parameters(), lr=lr_1, momentum=0.9)
        optimizer_2 = optim.SGD(self.mymodel.fc.parameters(), lr=lr_2, momentum=0.9)
        scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=lr_step_size, gamma=0.1)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=lr_step_size, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # training
        for epoch in range(epoch_nums):
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
            save_model_path= model_path +'model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.state_dict(),save_model_path)

    def test_network_aug_segment_2(self,pre_model = None, data_aug = 'aug_seg'):
        '''
        :param data_aug: 'aug_seg':   semantic augment 
        :param data_aug  'aug_seg_T:' semantic + temporal consistence augment
        :param data_aug  'aug_seg_random:' random select
        :param data_aug  'aug_seg_T_seg_len_1:' aug_seg_T + seg_len=1
        :param data_aug  ‘aug_seg_T_seg_len_4' aug_seg_T + seg_len=4
        :param data_aug  'aug_image_gaussian' use test_network_baseline + aug_video
        :param data_aug  'aug_frame_gaussian' use test_network_baseline + aug_video_feature
        '''

        if pre_model:
            self.mymodel.load_state_dict(torch.load(pre_model))
            print(pre_model, 'loaded.')

        self.mymodel.eval()

        # init file
        self.acc_file = open(self.test_result_txt, "w")

        # get gallery_videos and gallery_videos
        gallery_videos = generate_gallery_videos()                         # [640,16,3,224,224]
        gallery_segments = gallery_videos.view(-1, seg_len, 3, 224, 224)   # [640*8,2,3,224,224]
        gallery_videos = gallery_videos.view(-1, 3, 224, 224)              # [640*16,3,224,224]
        gallery_features = self.generate_epoch_features_2(gallery_videos, self.L2)  # [640*16,2048]
        gallery_seg_features = np.resize(gallery_features,(640 * VIDEO_FRAMES // seg_len, seg_len, 2048))  # [640*8,2,2048]
        gallery_seg_features = np.mean(gallery_seg_features, axis=1)       # [640*8,2048]
        # print('g_video:',gallery_videos.shape,'g_seg:',gallery_segments.shape,'g_fea:',gallery_seg_features.shape)

        epoch_nums = test_episodes
        accs = []
        num_segs = VIDEO_FRAMES // seg_len
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader_test.get_episode()
            support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
            query_features = self.generate_epoch_features(query_x, self.L2)
            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()
            support_segments = support_x.view(-1, seg_len, 3, 224, 224)   # [n_way*k_shot,16,3,224,224]   --> [n_way*k_shot*8,2,3,224,224]
            support_videos = support_x.view(-1, 3, 224, 224)              # [n_way*k_shot*16,3,224,224]
            support_features = self.generate_epoch_features_2(support_videos, self.L2)  # [n_way*k_shot*16,2048]
            support_seg_features = np.resize(support_features, (n_way*k_shot*VIDEO_FRAMES // seg_len, seg_len, 2048))  # [n_way*k_shot*8,2,2048]
            support_seg_features = np.mean(support_seg_features, axis=1)  # [n_way*k_shot*8,2048]

            # support_seg_features = self.generate_epoch_features(support_segments, self.L2)
            # print('support_x:',support_x.shape,'s_seg:',support_segments.shape,'s_fea:',support_seg_features.shape)
            # get seg_distance
            if (data_aug=='aug_seg'):
                distance = cdist(support_seg_features, gallery_seg_features, 'euclidean')
                # choose aim gallery_seg for per support_seg
                gallery_pool_ids = np.argsort(distance, axis=1)
                gallery_pool_ids = gallery_pool_ids[:, :1]
                # print('gallery_pool_ids:',gallery_pool_ids)
                gallery_pool_ids = np.resize(gallery_pool_ids, (n_way * k_shot, num_segs))
            elif (data_aug =='aug_seg_T' or data_aug == 'aug_seg_T_seg_len_1'or data_aug=='aug_seg_T_seg_len_4'):
                distance = cdist(support_seg_features, gallery_seg_features, 'euclidean')
                distance = self.temporal_convolution_flating_layer(distance)
                # choose aim gallery_seg for per support_seg
                gallery_pool_ids = np.argsort(distance, axis=1)
                gallery_pool_ids = gallery_pool_ids[:, :1]
                # print('gallery_pool_ids:',gallery_pool_ids)
                gallery_pool_ids = np.resize(gallery_pool_ids, (n_way * k_shot, num_segs))
            elif ( data_aug == 'aug_seg_random' or data_aug=='aug_image_gaussian' or data_aug =='aug_frame_gaussian'):
                probe_seg_nums = n_way * k_shot * num_segs
                gallery_pool_ids = self.get_gallery_pool_ids_random(probe_seg_nums)
                gallery_pool_ids = np.array(gallery_pool_ids)
                gallery_pool_ids = np.resize(gallery_pool_ids,(n_way*k_shot,num_segs))
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
                    
                    if(data_aug=='aug_frame_gaussian'):
                        video_aug_feature[seg_id*seg_len:seg_id*seg_len+seg_len,:] += np.random.normal(0,0.3,size=(seg_len,2048))

                    video_aug_feature = np.mean(video_aug_feature, axis=0)
                    # print('video_aug:',video_aug.shape,'video_aug_feature:',video_aug_feature.shape,'video_label:',video_label)
                    aug_video_features.append(video_aug_feature)
                    aug_video_labels.append(video_label)
            aug_video_features = np.array(aug_video_features)
            aug_video_labels = np.array(aug_video_labels)
            # print('aug_video_features:',aug_video_features.shape,'aug_video_labels:',aug_video_labels.shape)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result[
                'query_y'] = aug_video_features, aug_video_labels, query_features, query_y
            # print(data_result['support_feature'].shape, data_result['support_y'].shape,data_result['query_feature'].shape, data_result['query_y'].shape)

            # get query_feature
            # query_features=[]
            # for video_query in query_x:
            #     video_query = torch.FloatTensor(video_query)
            #     video_query = Variable(video_query).cuda()
            #     video_query_feature, output = self.mymodel(video_query)
            #     # add L2 norm
            #     video_query_feature = torch.nn.functional.normalize(video_query_feature, p=2, dim=1)
            #     video_query_feature = video_query_feature.cpu().detach().numpy()
            #     query_features.append(video_query_feature)
            # query_features = np.array(query_features)
            # print('query_video_features:',query_features.shape,'query_labels:',query_y.shape)

            # use aug_video_features and aug_video_labels to learn one-shot classifier
            # 1. protoNet
            # 2. SVM
            # 3. KNN
            # 4. logistic regression
            # 5. classifier NN
            if (self.classifier == 'protonet'):
                loss, acc = one_shot_classifier_prototype_lowerdim(data_result)

            elif (self.classifier == 'SVM'):
                classifier_SVM = SVC(C=10)  # TODO CHOOSE C AND OTHER PARAMS
                classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
                predicted_y = classifier_SVM.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0
            elif (self.classifier == 'KNN'):
                classifier_KNN = KNeighborsClassifier(n_neighbors=k_shot)
                classifier_KNN.fit(data_result['support_feature'], data_result['support_y'])
                predicted_y = classifier_KNN.predict(data_result['query_feature'])
                acc = np.mean(query_y == predicted_y)
                loss = 0

            else:
                print('classifier type error.')

            # test query ,get loss and acc
            print('epoch:', epoch,                'acc:', acc,'avg_acc:',np.mean(accs))
            print('epoch:', epoch, 'loss:', loss, 'acc:', acc,'avg_acc:',np.mean(accs), file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)

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

    def generate_trainAug_datasets_2_2(self, train_info = TRAIN_LIST, trainAug_dir = TrainAugSegDatasets_DIR_2_3):
        '''
        :param data_aug = 'aug_seg_T` default is for trainAug2.3
        '''
        train_video_list = open(train_info).readlines()

        # get gallery_videos and gallery_seg_videos and gallery_seg_features
        gallery_videos, gallery_videos_frames_dir = generate_gallery_videos_2()  # tensor [640,16,3,224,224]  # array[640,16]
        gallery_videos_frames_dir = np.resize(gallery_videos_frames_dir, (640 * 16))  # [640*16]

        # gallery_segments = gallery_videos.view(-1, seg_len, 3, 224, 224)  # tensor [640*8,2,3,224,224]
        # gallery_seg_features = self.generate_epoch_features(gallery_segments, self.L2)  # numpy [640*8,2048]
        gallery_videos = gallery_videos.view(-1, 3, 224, 224)  # []
        gallery_features = self.generate_epoch_features_2(gallery_videos, self.L2)  # [640*16,2048]
        gallery_seg_features = np.resize(gallery_features,(640 * VIDEO_FRAMES // seg_len, seg_len, 2048))  # [640*8,2,2048]
        gallery_seg_features = np.mean(gallery_seg_features, axis=1)
        # print(gallery_seg_features.shape)     # [640*8,2048]

        # print('gallery_videos:', gallery_videos.shape, 'gallery_videos_frames_dir', gallery_videos_frames_dir.shape)

        for i in range(len(train_video_list)):
        # for i in range(3000,4000):
            if(i==3000):
                i = 3850

            train_info = train_video_list[i].strip('\n')
            train_video, train_frames_dir = get_whole_video_from_video_info(train_info)
            # print(train_video.shape, train_frames_dir.shape,train_frames_dir[:2])    # tensor[300,3,224,224]  [300]

            norm_frames = train_video.shape[0] // seg_len * seg_len
            # print('video_frames:',train_video.shape[0],'norm_frames:',norm_frames)  # example 301, seq_len=2, norm =300

            train_video, train_frames_dir = train_video[:norm_frames], train_frames_dir[:norm_frames]
            # print('train_video:',train_video.shape,'train_frames_dir:',train_frames_dir.shape)

            # train_segments = train_video.view(-1,seg_len,3,224,224)     # tensor [150,2,3,224,224]
            # train_seg_features = self.generate_epoch_features(train_segments)  # [150,2048]

            train_features = self.generate_epoch_features_2(train_video, self.L2)  # [300,2048]
            train_seg_features = np.resize(train_features, (norm_frames // seg_len, seg_len, 2048))  # [150,2,2048]
            train_seg_features = np.mean(train_seg_features, axis=1)  # [150,2048]
            # print(train_seg_features.shape)  # [150,2048]

            distance = cdist(train_seg_features, gallery_seg_features, 'euclidean')

            # ADD new temporal convolution layer
            distance = self.temporal_convolution_flating_layer(distance)

            # print('distance:',distance.shape)                # [150 640*8]

            # choose aim gallery_seg for per support_seg
            gallery_pool_ids = np.argsort(distance, axis=1)  # [150,640*8]
            gallery_pool_ids = gallery_pool_ids[:, :1]  # [150,1]

            # get aug_video_frames and save them in new dir
            for i in range(norm_frames):
                # change the frames
                if i % VIDEO_FRAMES == 0:
                    gallery_seg_id = gallery_pool_ids[i // seg_len]
                    for j in range(seg_len):
                        # '/DATACENTER/2/lovelyqian/Kinetics/Kinetics/miniKinetics_frames/blowing glass/pkakR3JSTuU_000006_000016/image_00122.jpg'
                        init_frame_path = train_frames_dir[i + j]
                        gallery_frame_path = gallery_videos_frames_dir[gallery_seg_id * seg_len + j][0]
                        # print(init_frame_path)
                        # print(gallery_frame_path)

                        init_frame_path_splits = init_frame_path.split('/')
                        # print(init_frame_path_splits)

                        class_name = init_frame_path_splits[-3]
                        video_name = init_frame_path_splits[-2]
                        image_name = init_frame_path_splits[-1]
                        dst_class_path = os.path.join(trainAug_dir, class_name)
                        dst_directory_path = os.path.join(dst_class_path, video_name)
                        # if not os.path.exists(dst_class_path):
                        #     os.mkdir(dst_class_path)
                        # if not os.path.exists(dst_directory_path):
                        #     os.mkdir(dst_directory_path)
                        cmd = 'cp "%s" "%s/%s"' % (gallery_frame_path, dst_directory_path, image_name)
                        print(i, cmd)
                        subprocess.call(cmd, shell=True)
                elif i % VIDEO_FRAMES < seg_len:
                    pass
                else:
                    pass
                    # init_frame_path = train_frames_dir[i]
                    # # print(init_frame_path)
                    #
                    # init_frame_path_splits = init_frame_path.split('/')
                    # # print(init_frame_path_splits)
                    #
                    # class_name = init_frame_path_splits[-3]
                    # video_name = init_frame_path_splits[-2]
                    # image_name = init_frame_path_splits[-1]
                    # dst_class_path = os.path.join(trainAug_dir, class_name)
                    # dst_directory_path = os.path.join(dst_class_path, video_name)
                    # if not os.path.exists(dst_class_path):
                    #     os.mkdir(dst_class_path)
                    # if not os.path.exists(dst_directory_path):
                    #     os.mkdir(dst_directory_path)
                    # cmd = 'cp "%s" "%s"'%(init_frame_path, dst_directory_path)
                    # subprocess.call(cmd, shell=True)
                    # print(cmd)

    # AUGMENT DATA BY RANDOM SELECT
    def get_gallery_pool_ids_random(self,probe_seg_nums):
        gallery_seg_nums = 64 * 10 // seg_len
        aim_galllery_pool_ids = []
        for i in range(probe_seg_nums):
            aim_id = np.random.randint(gallery_seg_nums)
            aim_galllery_pool_ids.append(aim_id)
        return aim_galllery_pool_ids

    def generate_trainAug_datasets_2_4(self, train_info = TRAIN_LIST, trainAug_dir = TrainAugSegDatasets_DIR_2_4):
        '''
        :param data_aug = 'random` dataset2.4
        '''
        train_video_list = open(train_info).readlines()

        # get gallery_videos and gallery_seg_videos and gallery_seg_features
        gallery_videos, gallery_videos_frames_dir = generate_gallery_videos_2()  # tensor [640,16,3,224,224]  # array[640,16]
        gallery_videos_frames_dir = np.resize(gallery_videos_frames_dir, (640 * 16))  # [640*16]

        # for i in range(len(train_video_list)):
        for i in range(len(train_video_list)):
            train_info = train_video_list[i].strip('\n')
            train_video, train_frames_dir = get_whole_video_from_video_info(train_info)
            norm_frames = train_video.shape[0] // seg_len * seg_len
            train_video, train_frames_dir = train_video[:norm_frames], train_frames_dir[:norm_frames]

            # get random gallery_pool_ids
            probe_gallery_nums = norm_frames // seg_len
            gallery_pool_ids = self.get_gallery_pool_ids_random(probe_gallery_nums)

            # get aug_video_frames and save them in new dir
            for i in range(norm_frames):
                # change the frames
                if i % VIDEO_FRAMES == 0:
                    gallery_seg_id = gallery_pool_ids[i // seg_len]
                    for j in range(seg_len):
                        # '/DATACENTER/2/lovelyqian/Kinetics/Kinetics/miniKinetics_frames/blowing glass/pkakR3JSTuU_000006_000016/image_00122.jpg'
                        init_frame_path = train_frames_dir[i + j]
                        gallery_frame_path = gallery_videos_frames_dir[gallery_seg_id * 2 + j]
                        print(gallery_seg_id,gallery_frame_path)
                        init_frame_path_splits = init_frame_path.split('/')

                        class_name = init_frame_path_splits[-3]
                        video_name = init_frame_path_splits[-2]
                        image_name = init_frame_path_splits[-1]
                        dst_class_path = os.path.join(trainAug_dir, class_name)
                        dst_directory_path = os.path.join(dst_class_path, video_name)
                        cmd = 'cp "%s" "%s/%s"' % (gallery_frame_path, dst_directory_path, image_name)
                        print(i, cmd)
                        subprocess.call(cmd, shell=True)
                elif i % VIDEO_FRAMES < seg_len:
                    pass
                else:
                    pass




def trainTest():
    # exp1.1
    acc_path = 'train.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path,'resnet50','KNN',True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, data_aug = './result/train_idle.txt', './result/exp_idle/',300, 6, 0.001,0.01,10,'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, data_aug)


if __name__ == '__main__':
    # usage  TODO for test model

    '''
    acc_path = './result/acc_exp2.2_2000_testBase_proto_1shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path,'resnet50','protonet',True)
    myTestBaselineNet.test_network_baseline(pre_model='./result/exp2.2/model6.pkl')

    acc_path = './result/acc_exp2.3_2000_testBase_proto_1shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path,'resnet50','protonet',True)
    myTestBaselineNet.test_network_baseline(pre_model='./result/exp2.3/model6.pkl')
    '''
    '''
    acc_path = './RESULT/acc_exp10_2000_testAug_proto_1shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    myTestBaselineNet.test_network_aug_segment_2(pre_model='./RESULT/exp10/model6.pkl',data_aug = 'aug_seg_T')
    '''
    # cuda2
     
    acc_path = './RESULT/acc_exp3_20000_testBase_proto_5shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    myTestBaselineNet.test_network_baseline(pre_model='./RESULT/exp3/model6.pkl')

    '''
    acc_path = './RESULT/acc_exp1.0_2000_testBase_KNN_3shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    myTestBaselineNet.test_network_baseline(pre_model=None)
    '''

    # cuda3
    '''  
    acc_path = './RESULT/acc_exp1.0_2000_testBase_SVM_2shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'SVM', True)
    myTestBaselineNet.test_network_baseline(pre_model=None)

    acc_path = './RESULT/acc_exp1.0_2000_testBase_KNN_2shot_v2.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    myTestBaselineNet.test_network_baseline(pre_model=None)
    '''
    '''
    acc_path = './RESULT/acc_exp1.0_2000_testBase_SVM_5shot_v2.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'SVM', True)
    myTestBaselineNet.test_network_baseline(pre_model=None)
    '''
  


    
    
    # cuda0
    '''
    acc_path = './RESULT/acc_exp5_2000_testAug_proto_3shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    myTestBaselineNet.test_network_aug_segment_2(pre_model='./RESULT/exp5/model6.pkl',data_aug='aug_seg_T_seg_len_1')
    '''
   
 
    # cuda1
    ''' 
    acc_path = './RESULT/acc_exp3_2000_testAug_proto_3shot_v2.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    myTestBaselineNet.test_network_aug_segment_2(pre_model='./RESULT/exp3/model6.pkl',data_aug = 'aug_seg_T')
    '''
   

    '''  
    acc_path = './result/acc_exp7_2000_testAug_proto_4shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)   
    myTestBaselineNet.test_network_aug_segment_2(pre_model='./RESULT/exp7/model6.pkl',data_aug = 'aug_image_gaussian')
   

    acc_path = './result/acc_exp8_2000_testAug_proto_4shot.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    myTestBaselineNet.test_network_aug_segment_2(pre_model='./RESULT/exp8/model6.pkl',data_aug = 'aug_frame_gaussian')
    '''

    # usage  TODO for finetune model use Train dataset

    # exp1.1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path,'resnet50','KNN',True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, data_aug = './result/train1.1.txt', './result/exp1.1/',6, 6, 0.001,0.01,10,'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums,batch_size,lr_1, lr_2, lr_step_size, data_aug)
    '''

    # exp1.2
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.2.txt', './result/exp1.2/', 6, 6, 0.0001, 0.001, 10, 'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug)
    '''

    #new_exp1.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.3.txt', './result/exp1.3/', 6, 6, 0.0001, 0.001, 10, 'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model='./resultF/exp1.2/model6.pkl')
    '''

    # new_exp1.4i
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.4.txt', './result/exp1.4/', 6, 6, 0.00001, 0.0001, 10, 'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model='./resultF/exp1.2/model6.pkl')
    '''

    # exp1.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.3.txt', './result/exp1.3/', 6, 6, 0.001, 0.01, 10, 'aug_seg'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug)
    '''

    # exp1.4
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.4.txt', './result/exp1.4/', 6, 6, 0.0001, 0.001, 10, 'aug_seg'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug)
    '''
    
    # FINAL_2_EXP1.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # myTestBaselineNet.test_network_baseline()
    # myTestBaselineNet.test_network_aug_block()
    # myTestBaselineNet.test_network_aug_segment()
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug = './result/train1.3.txt', './RESULT/exp1.3/', 30, 6, 0.0001, 0.001, 10, 'None'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug)
    '''

    # usage TODO for aug_seg for train datasets V1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets()
    '''

    # usgae TODO for finetune model use trainAug DATA V1
    # exp1.5
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model = './result/train1.5.txt', './result/exp1.5/', 6, 6, 0.0001, 0.001, 10, 'aug_seg',None
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
    # exp1.6
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model = './result/train1.6.txt', './result/exp1.6/', 12, 6, 0.0001, 0.001, 6, 'aug_seg',None
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

     # exp1.7  
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model = './result/train1.7_true.txt', './result/exp1.7_true/', 6, 6, 0.0001, 0.001, 10, 'aug_seg','./result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # exp1.8
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model = './result/train1.8.txt', './result/exp1.8/', 12, 6, 0.0001, 0.001, 6 ,'aug_seg','./result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # usage  TODO for baseline_resnet50_cosine
    '''
    acc_path = './result/acc_baseline_resnet50_cosine.txt '
    myTestBaselineNet = TestBaselineNetwork(acc_path,'resnet50','',True)
    myTestBaselineNet.test_network_baseline(pre_model=None)
    '''

    # usage TODO FOR AUG_SEG_FOR ALL FRAMES V2.1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_1()
    '''

    # uasge TODO FOR FINETUNE MODEL WITH aug_seg_dateset V2.1
    # new_exp2
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train2.1.txt', './resultF/exp2/', 6, 6, 0.0001, 0.001, 6, 'aug_seg', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
    '''
    # new exp2.2
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train2.2.txt', './resultF/exp2.2/', 6, 6, 0.0001, 0.001, 6, 'aug_seg', './resultF/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
   
    # new exp2.3
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train2.3.txt', './resultF/exp2.3/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg', './resultF/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
    
    # FINAL_EXP2
    # frames_num=16, seg_len=2,aug_seg,dataset2.1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train2.txt', './RESULT/exp2/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
  
    # FINAL_2_EXP2.6
    # frames_num=16, seg_len=2,aug_seg,dataset2.1
    ''' 
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train2.6.txt', './RESULT/exp2.6/', 30, 6, 0.00001, 0.0001, 10, 'aug_seg', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''


    # exp1.9
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train1.9.txt', './result/exp1.9/', 6, 6, 0.0001, 0.001, 6, 'aug_seg', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # exp1.10
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train1.10.txt', './result/exp1.10/', 6, 6, 0.0001, 0.001, 6, 'aug_seg', './result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''

    # exp1.11
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train1.10.txt', './result/exp1.10/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg', './result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''

    # rebuttal exp9 semantic closest frames_2.1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train9.txt', './RESULT/exp9/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''

    # usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.2
    # kernal =[0.6,1,0.6]
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2()
    '''

    # rebuttal exp10 [0.6,1.0,0.6] frames_2.2
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train10.txt', './RESULT/exp10/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T_0.6', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''



    # usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.3
    # kernal =[0.1,1,0.1]
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2()
    '''



    # rebuttal usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.8
    # kernal =[1.0,1,1.0]
    ''' 
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2(trainAug_dir = TrainAugSegDatasets_DIR_2_8)
    '''

    # rebuttal usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.9
    # kernal =[2.0,1,2.0]
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2(trainAug_dir = TrainAugSegDatasets_DIR_2_9)
    '''

    # rebuttal usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.10
    # kernal =[5.0,1,5.0]]
    # ONLY ONCE
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'KNN', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2(trainAug_dir = TrainAugSegDatasets_DIR_2_10)
    '''

    # uasge TODO FOR FINETUNE MODEL WITH aug_seg_dateset V2.3
    # exp2.1
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train2.1.txt', './result/exp2.1/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_T', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # exp2.2
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train2.2.txt', './result/exp2.2/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_T', './result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''

    # exp2.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train2.3.txt', './result/exp2.3/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T', './result/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size,data_aug, pre_model)
    '''

    '''

    # new_exp3.1
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train3.1.txt', './resultF/exp3.1/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_T', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    
    
    # new exp3.2
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train3.2.txt', './resultF/exp3.2/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_T', './resultF/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)

    # new exp3.3
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './resultF/train3.3.txt', './resultF/exp3.3/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T', './resultF/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_EXP3
    # frames_num=16, seg_len=2,aug_seg_T,dataset2.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.txt', './RESULT/exp3/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
   
    '''
    # FINAL_2_EXP3.2
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.2.txt', './RESULT/exp3.2/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
    # FINAL_2_EXP3.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.3.txt', './RESULT/exp3.3/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_T', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_2_EXP3.4
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.4.txt', './RESULT/exp3.4/', 6, 6, 0.000001, 0.00001, 6, 'aug_seg_T', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
   # FINAL_2_EXP3.5
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.5.txt', './RESULT/exp3.5/', 6, 6, 0.001, 0.01, 6, 'aug_seg_T', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_2_EXP3.6
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.6.txt', './RESULT/exp3.6/', 30, 6, 0.0001, 0.001, 10, 'aug_seg_T',None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_EXP3.2
    # frames_num=16, seg_len=2,aug_seg_T,dataset2.3
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train3.2.txt', './RESULT/exp3.2/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # usage TODO FOR AUG_SEG_FOR ALL FRAMES ,random select V2.4
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_4()
    '''

    # rebuttal FOR AUG_SEG_FOR frames, add gaussian noise V2.7
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_7()
    '''

    # exp3.1
    '''  
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './result/train3.1.txt', './result/exp3.1/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_random', None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    ''' 

    # FINAL_EXP4
    # frames_num=16, seg_len=2,aug_seg_random,dataset2.4
    '''  
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train4.txt', './RESULT/exp4/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_random', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''


    # FINAL_2_EXP4.2
    # frames_num=16, seg_len=2,aug_seg_random,dataset2.4
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train4.2.txt', './RESULT/exp4.2/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_random', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_2_EXP4.3
    # frames_num=16, seg_len=2,aug_seg_random,dataset2.4
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train4.3.txt', './RESULT/exp4.3/', 6, 6, 0.0001, 0.001, 6, 'aug_seg_random', './RESULT/exp1.3/model30.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
   
    # FINAL_2_EXP4.6
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train4.6.txt', './RESULT/exp4.6/', 30, 6, 0.0001, 0.001, 10, 'aug_seg_random',None
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.5 (V2.3 + seg_len = 1)
    # kernal =[0.1,1,0.1]
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2(trainAug_dir = TrainAugSegDatasets_DIR_2_5)
    '''
 
    # FINAL_EXP5
    # frames_num=16, seg_len=1,aug_seg_T_seg_len_1,dataset2.5
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train5.txt', './RESULT/exp5/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T_seg_len_1', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # usage TODO FOR AUG_SEG_FOR ALL FRAMES and ADD temporal consistency V2.6 (V2.3 + seg_len = 4)
    # kernal =[0.1,1,0.1]
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    # ONLY ONCE
    myTestBaselineNet.generate_trainAug_datasets_2_2(trainAug_dir = TrainAugSegDatasets_DIR_2_6)
    '''

    # FINAL_EXP6
    # frames_num=16, seg_len=4, aug_seg_T_seg_len_4,dataset2.6
    '''
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train6.txt', './RESULT/exp6/', 6, 6, 0.00001, 0.0001, 6, 'aug_seg_T_seg_len_4', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''



    # FINAL_EXP7
    # frames_num=16, seg_len=2, aug_image_gaussian,dataset_INIT
    ''' 
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train7.txt', './RESULT/exp7/', 6, 6, 0.00001, 0.0001, 6, 'aug_image_gaussian', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''

    # FINAL_EXP8
    # frames_num=16, seg_len=2. aug_frame_gaussian,dataset_INIT
    '''    
    acc_path = 'test.txt'
    myTestBaselineNet = TestBaselineNetwork(acc_path, 'resnet50', 'protonet', True)
    loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug, pre_model = './RESULT/train8.txt', './RESULT/exp8/', 6, 6, 0.00001, 0.0001, 6, 'aug_frame_gaussian', './RESULT/exp1.2/model6.pkl'
    myTestBaselineNet.finetune_model_2(loss_path, model_path, epoch_nums, batch_size, lr_1, lr_2, lr_step_size, data_aug,pre_model)
    '''
   
    #trainTest()

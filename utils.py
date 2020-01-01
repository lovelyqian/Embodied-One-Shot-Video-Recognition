import os
import cv2
import copy
import torch
import pickle
import random
import numpy as np
import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional


# global path
KINETICS_VIDEO_DIR = '/DATACENTER/2/lovelyqian/Kinetics/Kinetics/videos/'
KINETICS_FRAME_DIR = '/DATACENTER/2/lovelyqian/Kinetics/Kinetics/miniKinetics_frames/'
#INIT_TRAIN_LIST = 'sources/data/init_train.list'
#INIT_VAL_LIST = 'sources/data/init_val.list'
#INIT_TEST_LIST = 'sources/data/init_test.list'
TRAIN_LIST = 'sources/data/train.list'
VAL_LIST = 'sources/data/val.list'
TEST_LIST = 'sources/data/test.list'

GALLERY_LIST = '../sources/data/gallery.list'
#TrainAugSegDatasets_DIR_2_1 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.1/'  # semantic closest
#TrainAugSegDatasets_DIR_2_2 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.2/'  # semantic closest + temporal consistence [0.6,1,0.6]
TrainAugSegDatasets_DIR_2_3 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.3/'  # semantic closest + temporal consistence [0.1,1,0.1]
#TrainAugSegDatasets_DIR_2_4 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.4/'  # random select
#TrainAugSegDatasets_DIR_2_5 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.5/'  # 2.3 + seg_len=1
#TrainAugSegDatasets_DIR_2_6 = '/DATACENTER/s/lovelyqian/miniKinetics_frames_2.6/'  # 2.3 + seg_len=4

# gloable variable
num_classes_train = 64
VIDEO_FRAMES = 16
IMG_INIT_H=256
IMG_crop_size = (224,224)
BATCH_SIZE=6
n_way = 5
k_shot = 1
seg_len=2
test_episodes = 20000
val_episodes = 100
lamda1, lamda2 = 0.1, 1.0
EPISODE_NUMS = {'test': test_episodes, 'val': val_episodes}

#gloable functions
def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        #if (not k in wf)):
        if ((not k in wf) | (k=='fc.weight') | (k=='fc.bias')):
            wf[k] = wt[k]
    model_to.load_state_dict(wf)

# for Video Processing
class ClipRandomCrop(torchvision.transforms.RandomCrop):
  def __init__(self, size):
    self.size = size
    self.i = None
    self.j = None
    self.th = None
    self.tw = None

  def __call__(self, img):
    if self.i is None:
      self.i, self.j, self.th, self.tw = self.get_params(img, output_size=self.size)
    return torchvision.transforms.functional.crop(img, self.i, self.j, self.th, self.tw)

class ClipRandomHorizontalFlip(object):
  def __init__(self, ratio=0.5):
    self.is_flip = random.random() < ratio

  def __call__(self, img):
    if self.is_flip:
      return torchvision.transforms.functional.hflip(img)
    else:
      return img

def transforms(mode):
    if (mode=='train'):
        random_crop = ClipRandomCrop(IMG_crop_size)
        flip = ClipRandomHorizontalFlip(ratio=0.5)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([random_crop, flip,toTensor,normalize])
    else:   # mode=='test'
        center_crop = torchvision.transforms.CenterCrop(IMG_crop_size)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([center_crop,toTensor,normalize])



# the basic , only can be used when video_frames=16 , (frames_num>16, return video frames unequal
def get_video_from_video_info(video_info,mode,video_frames= VIDEO_FRAMES,frame_dir = KINETICS_FRAME_DIR,data_aug=None):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :param mode:  train{random continuous 32-clip random-crop} , test{middle continuous 32-clip center-crop}
    :return: torch.Size([16, 3, 242, 242])
    '''
    video_frame_path = os.path.join(frame_dir,video_info)
    all_frame_count = len(os.listdir(video_frame_path))-1

    if(all_frame_count -video_frames-1 >1):
        if (mode == 'train'):
            image_start = random.randint(1, all_frame_count - video_frames -1)
        # get middle 32-frame clip
        elif ((mode == 'test') | (mode=='val')):
            image_start = all_frame_count // 2 -video_frames // 2 + 1
    else:
        image_start=1   # use 0 padding

    image_id = image_start
    myTransform = transforms(mode=mode)
    video=[]
    for i in range(video_frames):
        s = "%05d" % image_id
        image_name = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_name)
        image = Image.open(image_path)
        
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image = myTransform(image)
        video.append(image)

        image_id += 1
        if (image_id > all_frame_count):
            break

    video=torch.stack(video,0)
    
   
    return video



def get_classname_from_video_info(video_info):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :return: classnum:air drumming
    '''
    video_info_splits = video_info.split('/')
    class_num = video_info_splits[0]
    return class_num

# global functions
def get_classInd(info_list):
    info_list=open(info_list).readlines()
    classlabel=0
    classInd={}
    for info_line in info_list:
        info_line=info_line.strip('\n')
        videoname= get_classname_from_video_info(info_line)
        if videoname not in classInd.keys():
            classInd[videoname]=classlabel
            classlabel = classlabel +1
        else:
            pass
    return classInd

def get_label_from_video_info(video_info,info_list = TRAIN_LIST):
   classname = get_classname_from_video_info(video_info)
   classInd = get_classInd(info_list)
   label = classInd[classname]
   return label

# return video, np.array(frames_dir)
def get_video_from_video_info_2(video_info,mode,video_frames= VIDEO_FRAMES,frame_dir = KINETICS_FRAME_DIR):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :param mode:  train{random continuous 32-clip random-crop} , test{middle continuous 32-clip center-crop}
    :return: torch.Size([16, 3, 242, 242])
    '''
    video_frame_path = os.path.join(frame_dir,video_info)
    all_frame_count = len(os.listdir(video_frame_path))-1

    if(all_frame_count - video_frames -1  >1):
        if (mode == 'train'):
            image_start = random.randint(1, all_frame_count - video_frames -1)
        # get middle 32-frame clip
        elif ((mode == 'test') | (mode=='val')):
            image_start = all_frame_count // 2 -video_frames // 2 + 1
    else:
        image_start=1   # use 0 padding

    image_id = image_start
    myTransform = transforms(mode=mode)
    video=[]
    frames_dir = []
    for i in range(video_frames):
        s = "%05d" % image_id
        image_name = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_name)
        image = Image.open(image_path)
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image = myTransform(image)
        video.append(image)

        frames_dir.append(image_path)

        image_id += 1
        if (image_id > all_frame_count):
            break

    video=torch.stack(video,0)
    return video,np.array(frames_dir)


# return video, np.array(frames_num)
def get_video_from_video_info_3(video_info,mode,video_frames= VIDEO_FRAMES,frame_dir = KINETICS_FRAME_DIR):
    '''
    :param video_info: air drumming/-VtLx-mcPds_000012_000022
    :param mode:  train{random continuous 32-clip random-crop} , test{middle continuous 32-clip center-crop}
    :return: torch.Size([16, 3, 242, 242])
    '''
    video_frame_path = os.path.join(frame_dir,video_info)
    all_frame_count = len(os.listdir(video_frame_path))-1

    if(all_frame_count - video_frames -1  >1):
        if (mode == 'train'):
            image_start = random.randint(1, all_frame_count - video_frames -1)
        # get middle 32-frame clip
        elif ((mode == 'test') | (mode=='val')):
            image_start = all_frame_count // 2 -video_frames // 2 + 1
    else:
        image_start=1   # use 0 padding

    image_id = image_start
    myTransform = transforms(mode=mode)
    video=[]
 
    for i in range(video_frames):
        s = "%05d" % image_id
        image_name = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_name)
        image = Image.open(image_path)
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image = myTransform(image)
        video.append(image)

        image_id += 1
        if (image_id > all_frame_count):
            for j in range(image_id,video_frames+1):
                image = torch.zeros((3,224,224))
                video.append(image)
            break

    video=torch.stack(video,0)
    
    video_frame_count = np.min((VIDEO_FRAMES,all_frame_count))
    return video, video_frame_count




if __name__=='__main__':
    video_info = 'air drumming/-VtLx-mcPds_000012_000022'


    video = get_video_from_video_info(video_info,mode='train')
    print(video.shape)

    label = get_label_from_video_info(video_info)
    
    '''
    from baseline_resnet_pretrained import trainTest
    trainTest()
    '''
    
    print('video frames:',VIDEO_FRAMES)   
    test_info = open(TEST_LIST).readlines()
    for i in range(len(test_info)):
        test_list = test_info[i]
        test_list = test_list.strip('\n')
        video,video_frame_num = get_video_from_video_info_3(test_list,mode='test')
        print(i,video.shape,video.shape[0]==VIDEO_FRAMES,video_frame_num)
        
        ''' 
        if(video_frame_num<VIDEO_FRAMES):
            print(video[0],video[video_frame_num])
        '''

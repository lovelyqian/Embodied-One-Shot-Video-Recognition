import numpy as np
from utils import *

def generate_gallery_list(num=10):
    '''
    from train.list  generate gallery.list (random 10 videos/class)
    '''
    data = open(TRAIN_LIST).readlines()
    file = open(GALLERY_LIST,"w")
    dict = {}
    for line in data:
        line = line.strip('\n')
        class_num = line.split('/')[0]
        if class_num not in dict.keys():
            dict[class_num] = [line]
        else:
            dict[class_num].append(line)

    for class_num in dict.keys():
        info_list = dict[class_num]
        aim_info_list = random.sample(info_list,num)
        for aim_info in aim_info_list:
            print(aim_info,file=file)

def generate_gallery_videos():
    '''
    :return: gallery_videos (FRAME_NUMS_def, mode=test)
    '''
    videos=[]
    data = open(GALLERY_LIST).readlines()
    for line in data:
        video_info = line.strip('\n')
        video = get_video_from_video_info(video_info,mode='test')
        videos.append(video)
    videos = torch.stack(videos)
    return videos

def generate_gallery_videos_2():
    '''
    :return: gallery_videos (FRAME_NUMS_def, mode=test)
    '''
    videos=[]
    video_frames_dir =[]
    data = open(GALLERY_LIST).readlines()
    for line in data:
        video_info = line.strip('\n')
        video,frames_dir = get_video_from_video_info_2(video_info,mode='test')
        video_frames_dir.append(frames_dir)
        videos.append(video)
    videos = torch.stack(videos)
    video_frames_dir = np.array(video_frames_dir)
    return videos,video_frames_dir


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



def temporal_convolution_flating_layer(distance):
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



def generate_trainAug_datasets(train_info = TRAIN_LIST, trainAug_dir = TrainAugSegDatasets_DIR_2_3):
    '''
    :param data_aug = 'aug_seg_T` default is for trainAug2.3
    '''
    train_video_list = open(train_info).readlines()

    # get gallery_videos and gallery_seg_videos and gallery_seg_features
    gallery_videos, gallery_videos_frames_dir = generate_gallery_videos_2()  # tensor [640,16,3,224,224]  # array[640,16]
    gallery_videos_frames_dir = np.resize(gallery_videos_frames_dir, (640 * 16))  # [640*16]

    gallery_videos = gallery_videos.view(-1, 3, 224, 224)  # []
    gallery_features = generate_epoch_features_2(gallery_videos, self.L2)  # [640*16,2048]
    gallery_seg_features = np.resize(gallery_features,(640 * VIDEO_FRAMES // seg_len, seg_len, 2048))  # [640*8,2,2048]
    gallery_seg_features = np.mean(gallery_seg_features, axis=1)


    for i in range(len(train_video_list)):
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




if __name__=='__main__':
    # ONLY ONCE
    generate_gallery_list()
    generate_trainAug_datasets()

import torch
from utils import  *

class EpisodeDataloader():
    '''
    get_episode: return episode
    shuffle label every episode
    '''
    def __init__(self,mode='test'):
        self.mode = mode
        if(mode == 'train'):
            self.dataset_list = TRAIN_LIST
        elif(mode == 'val'):
            self.dataset_list = VAL_LIST
        elif(mode == 'test'):
            self.dataset_list = TEST_LIST
        self.data = open(self.dataset_list).readlines()

    def get_episode(self):
        '''
        :return: support_x = n_way * k_shot * video, support_y = n_way * k_shot * y,;
        :return: query_x = 1* video , query_y = 1 * y
        '''
        # handle dataset_info
        dict = {}
        for line in self.data:
            line = line.strip('\n')
            class_name = line.split('/')[0]
            if class_name not in dict.keys():
                dict[class_name] = [line]
            else:
                dict[class_name].append(line)

        # sample n-way class_name and shuffle then
        aim_class_names = random.sample(dict.keys(),n_way)
        # sample 1 query_name
        aim_query_name = random.sample(aim_class_names,1)[0]

        # sample n-way * k_shot support sets  and 1 quey video
        support_x = []
        support_x_frames=[]
        support_y = []
        query_x = []
        query_y = []
        for class_name in aim_class_names:
            # get the additional one for query
            if (class_name == aim_query_name):
                aim_video_infos = random.sample(dict[class_name], k_shot +1 )
                video_info = aim_video_infos[0]
                # query is always mode='test'
                video = get_video_from_video_info(video_info,mode='test')
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                query_x.append(video)
                query_y.append(video_y)
                aim_video_infos=aim_video_infos[1:]
            else:
                aim_video_infos = random.sample(dict[class_name],k_shot)
            # sample support set
            for video_info in aim_video_infos:
                # support depends on it's mode
                # video = get_video_from_video_info(video_info,mode=self.mode)
            
                video,video_frame = get_video_from_video_info_3(video_info,mode=self.mode)
                support_x_frames.append(video_frame)
                
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                support_x.append(video)
                support_y.append(video_y)

        support_x = torch.stack(support_x)
        support_x = torch.FloatTensor(support_x)
        support_y = torch.FloatTensor(support_y)

        query_x = torch.stack(query_x)
        query_x = torch.FloatTensor(query_x)
        query_y = torch.FloatTensor(query_y)

        return ({'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y,'support_x_frames':support_x_frames})



if __name__=='__main__':
    # usage
    myEpisodeDataloader = EpisodeDataloader(mode='test')
    data = myEpisodeDataloader.get_episode()
    print(data['support_x'].shape,data['support_y'].shape,data['query_x'].shape,data['query_y'].shape)
    # torch.Size([5, 16, 3, 242, 242]) torch.Size([5]) torch.Size([1, 16, 3, 242, 242]) torch.Size([1])

    for i in range(10):
        data = myEpisodeDataloader.get_episode()



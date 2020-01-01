import torch
from torch.utils.data import Dataset, DataLoader
from utils import *


# dataloader for train
class VideoDataset(Dataset):
    def __init__(self, info_txt, root_dir, mode='train',data_aug=None,transform=None):
        # set params
        self.info_txt=info_txt
        self.root_dir=root_dir
        self.mode=mode
        self.data_aug=data_aug
        self.transform = transform

        # read info_list
        self.info_list=open(self.info_txt).readlines()

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info_line=self.info_list[idx]
        video_info=info_line.strip('\n')
        video=get_video_from_video_info(video_info,mode=self.mode,frame_dir=self.root_dir,data_aug=self.data_aug)
        video_label=get_label_from_video_info(video_info,self.info_txt)

        sample = {'video': video, 'label': [int(video_label)]}

        sample['video'] = torch.FloatTensor(sample['video'])
        sample['label'] = torch.FloatTensor(sample['label'])

        return sample

if __name__ == '__main__':
    # usage-train
    '''    
    train=VideoDataset(TRAIN_LIST,KINETICS_FRAME_DIR,mode='train')
    Dataloader_train = DataLoader(train, batch_size= 4, shuffle=True, num_workers=8)

    for i_batch, sample_batched in enumerate(Dataloader_train):
        print(i_batch, sample_batched['video'].size(), sample_batched['label'].size())
        # torch.Size([4, 16, 3, 242, 242]) torch.Size([4, 1])
    '''

    # usage-test
    testDataset = VideoDataset(TEST_LIST,KINETICS_FRAME_DIR,mode='test')
    Dataloader_test = DataLoader(testDataset, batch_size= 6, shuffle=False, num_workers=8)

    for i_batch, sample_batched in enumerate(Dataloader_test):
        print(i_batch, sample_batched['video'].size(), sample_batched['label'].size(),sample_batched['label'])
        # torch.Size([4, 16, 3, 242, 242]) torch.Size([4, 1])


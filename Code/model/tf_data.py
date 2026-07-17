import os
import csv
import torch
import random
import numpy as np
from torch.autograd import Variable
from model.tf_model import subsequent_mask
from torch.utils.data import Dataset


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, ys=None, pad=0):
        self.src = src
        self.ys = ys
        self.src_len = src[:, 0, :]
        self.src_mask = (self.src_len != pad).unsqueeze(-2)
        if trg is not None:
            # ------for EEG-------
            self.trg = trg[:, :, :-1]
            self.trg_len = trg[:, 0, :]
            self.trg_x = self.trg_len[:, :-1]
            self.trg_y = self.trg_len[:, 1:]
            # ------for NLP-------
            #self.trg = trg[:, :-1]
            #self.trg_x = trg[:, :-1]
            #self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg_x, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

def data_load(train_loader):
    print("data_load:", len(train_loader))
    for i, (attr, target, ys) in enumerate(train_loader):
        #src = Variable(attr, requires_grad=False)
        #tgt = Variable(target, requires_grad=False)
        #print("data_load1:", attr.shape)
        #print("data_load2:", target.shape)
        src = attr.cuda()
        tgt = target.cuda()
        ys = ys.cuda()
        yield Batch(src, tgt, ys, 0)


class preDataset(Dataset):
    def __init__(self, mode, train_len=870300, block_num=1):
        self.sample_rate = 256
        self.lenth = 870300 #train_len
        self.lenthtest = 3600
        self.lenthval = 3500
        self.mode = mode
        self.block_num = block_num

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        data_mode = ["Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other"]
        ## step 1 locate idx into dataset
        dataset = ["Hyperscanning_navigation", "Hyperscanning_slapjack", "Lane_keeping"]

        if idx < 249816:
            now_idx = idx
            dataloc = 0
        elif idx >= 249816 and idx < 506365:
            now_idx = idx - 249816
            dataloc = 1
        elif idx >= 506365:
            now_idx = idx - 506365
            dataloc = 2

        ## step 2 read data
        folder_name = './MetaPreTrain/' + dataset[dataloc] + '/3_ICA/'
        allFileList = os.listdir(folder_name)
        file_name1 = folder_name + allFileList[now_idx]
        #print("preDataset1: ", allFileList[now_idx])
        ## get after 4 sec data
        string_array = allFileList[now_idx]
        parts = string_array[0].split('_')
        numeric_part = parts[-1].split('.')[0]
        new_numeric_part = str(int(numeric_part) + 4)
        parts[-1] = new_numeric_part + '.csv'
        string_array = '_'.join(parts)
        file_name2 = folder_name + string_array
        #print("preDataset2: ", string_array)
        try:
            data_nosie = self.read_train_data(file_name1)
            data_clean = self.read_train_data(file_name2)
        except:
            data_nosie = self.read_train_data(file_name1)
            data_clean = data_nosie

        ## step 3 signal normalize
        max_num = np.max(data_nosie)
        data_avg = np.average(data_nosie)
        data_std = np.std(data_nosie)

        if int(data_std) != 0:
            target = np.array((data_clean - data_avg) / data_std).astype(np.float64)
            attr   = np.array((data_nosie - data_avg) / data_std).astype(np.float64)
        else:
            target = np.array(data_clean - data_avg).astype(np.float64)
            attr   = np.array(data_nosie - data_avg).astype(np.float64)

        ## step 4 deep copy & return
        #target = target.copy()
        target = torch.FloatTensor(target)

        #attr = attr.copy()
        attr = torch.FloatTensor(attr)

        ys = torch.ones(30, 1023).fill_(1).type_as(attr)

        return attr, target, ys

    def read_train_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        new_data = np.array(data).astype(np.float64)

        return new_data


class myDataset(Dataset):
    def __init__(self, mode, train_len=0, block_num=1):
        self.sample_rate = 256
        self.lenth = train_len
        self.lenthtest = 3600
        self.lenthval = 3500
        self.mode = mode
        self.block_num = block_num

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        data_mode = ["Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other"]

        if self.mode == 2:
            allFileList = os.listdir("./Real_EEG/val/Brain/")
            file_name = './Real_EEG/val/Brain/' + allFileList[idx]
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/val/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        elif self.mode == 1:
            allFileList = os.listdir("./Real_EEG/test/Brain/")
            file_name = './Real_EEG/test/Brain/' + allFileList[idx]
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/test/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        else:
            allFileList = os.listdir("./Real_EEG/train/Brain/")
            file_name = './Real_EEG/train/Brain/' + allFileList[idx]
            #print("dataloader: ", file_name)
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/train/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        #print(file_name)



        #print("data_set", noise.shape)

        max_num = np.max(data_nosie)
        data_avg = np.average(data_nosie)
        data_std = np.std(data_nosie)
        #max_num = 100
        #print("max_num: ", max_num)

        #target = np.array(data / max_num).astype(np.float)
        if int(data_std) != 0:
            target = np.array((data_clean - data_avg) / data_std).astype(np.float64)
            attr   = np.array((data_nosie - data_avg) / data_std).astype(np.float64)
        else:
            target = np.array(data_clean - data_avg).astype(np.float64)
            attr   = np.array(data_nosie - data_avg).astype(np.float64)


        ## step 4 deep copy & return
        # target = target.copy()
        target = torch.FloatTensor(target)

        # attr = attr.copy()
        attr = torch.FloatTensor(attr)

        ys = torch.ones(30, 1023).fill_(1).type_as(attr)

        return attr, target, ys

    def read_train_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        new_data = np.array(data).astype(np.float64)

        ''' for training 19 channels
        row = np.array([0,1,2,3,4,5,6,12,13,14,15,16,22,23,24,25,26,27,29])
        new_data = []
        for i in range(19):
            #print(i, row[i])
            #print(data[row[i]].shape)
            new_data.append(data[row[i]])        
            new_data = np.array(new_data).astype(np.float)
        '''

        # data = data.T
        return new_data
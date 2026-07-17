import numpy as np
import csv
from model import cumbersome_model2
from model import UNet_family
from model import UNet_attention
from model import tf_model
from model import tf_data

import time
import torch
import os
import random
import shutil
from scipy.signal import decimate, resample_poly, firwin, lfilter


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def resample(signal, fs, tgt_fs):
    # downsample the signal to the target sample rate
    if fs>tgt_fs:
        fs_down = tgt_fs # Desired sample rate
        q = int(fs / fs_down) # Downsampling factor
        signal_new = []
        for ch in signal:
            x_down = decimate(ch, q)
            signal_new.append(x_down)

    # upsample the signal to the target sample rate
    elif fs<tgt_fs:
        fs_up = tgt_fs  # Desired sample rate
        p = int(fs_up / fs)  # Upsampling factor 
        signal_new = []
        for ch in signal:
            x_up = resample_poly(ch, p, 1)
            signal_new.append(x_up)

    else:
        signal_new = signal

    signal_new = np.array(signal_new).astype(np.float64)

    return signal_new

def FIR_filter(signal, lowcut, highcut):
    fs = 256.0
    # Number of FIR filter taps
    numtaps = 1000
    # Use firwin to create a bandpass FIR filter
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    # Apply the filter to signal:
    filtered_signal  = lfilter(fir_coeff, 1.0, signal)
    
    return filtered_signal


def read_train_data(file_name):
    with open(file_name, 'r', newline='') as f:
        lines = csv.reader(f)
        data = []
        for line in lines:
            data.append(line)

    data = np.array(data).astype(np.float64)
    return data


def cut_data(filepath, raw_data):
    raw_data = np.array(raw_data).astype(np.float64)
    total = int(len(raw_data[0]) / 1024)
    for i in range(total):
        table = raw_data[:, i * 1024:(i + 1) * 1024]
        filename = filepath + 'temp2/' + str(i) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)
    return total


def glue_data(file_name, total):
    gluedata = 0
    for i in range(total):
        file_name1 = file_name + 'output{}.csv'.format(str(i))
        with open(file_name1, 'r', newline='') as f:
            lines = csv.reader(f)
            raw_data = []
            for line in lines:
                raw_data.append(line)
        raw_data = np.array(raw_data).astype(np.float64)
        #print(i)
        if i == 0:
            gluedata = raw_data
        else:
            smooth = (gluedata[:, -1] + raw_data[:, 1]) / 2
            gluedata[:, -1] = smooth
            raw_data[:, 1] = smooth
            gluedata = np.append(gluedata, raw_data, axis=1)
    #print(gluedata.shape)
    return gluedata


def save_data(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def dataDelete(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        pass
        #print(e)
    else:
        pass
        #print("The directory is deleted successfully")


def decode_data(data, std_num, mode=5):
    
    if mode == "ICUNet":
        # 1. read name
        model = cumbersome_model2.UNet1(n_channels=30, n_classes=30).to(device)
        resumeLoc = './model/ICUNet/modelsave' + '/checkpoint.pth.tar'
        # 2. load model
        checkpoint = torch.load(resumeLoc, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], False)
        model.eval()
        # 3. decode strategy
        with torch.no_grad():
            data = data[np.newaxis, :, :]
            data = torch.Tensor(data).to(device)
            decode = model(data)

      
    elif mode == "ICUNet++" or mode == "ICUNet_attn":
        # 1. read name
        if mode == "ICUNet++":
            model = UNet_family.NestedUNet3(num_classes=30).to(device)
        elif mode == "ICUNet_attn":
            model = UNet_attention.UNetpp3_Transformer(num_classes=30).to(device)
        resumeLoc = './model/' + mode + '/modelsave' + '/checkpoint.pth.tar'
        # 2. load model
        checkpoint = torch.load(resumeLoc, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], False)
        model.eval()
        # 3. decode strategy
        with torch.no_grad():
            data = data[np.newaxis, :, :]
            data = torch.Tensor(data).to(device)
            decode1, decode2, decode = model(data)
            

    elif mode == "ART":
        # 1. read name
        resumeLoc = './model/' + mode + '/modelsave/checkpoint.pth.tar'
        # 2. load model
        checkpoint = torch.load(resumeLoc, map_location=device)
        model = tf_model.make_model(30, 30, N=2).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        # 3. decode strategy
        with torch.no_grad():
            data = torch.FloatTensor(data).to(device)
            data = data.unsqueeze(0)
            src = data
            tgt = data # you can modify to randomize data
            batch = tf_data.Batch(src, tgt, 0)
            out = model.forward(batch.src, batch.src[:,:,1:], batch.src_mask, batch.trg_mask)
            decode = model.generator(out)
            decode = decode.permute(0, 2, 1)
            add_tensor = torch.zeros(1, 30, 1).to(device)
            decode = torch.cat((decode, add_tensor), dim=2)

    # 4. numpy
    #print(decode.shape)
    decode = np.array(decode.cpu()).astype(np.float64)
    return decode


def reorder_data(raw_data, mapping_result):
    new_data = np.zeros((30, raw_data.shape[1]))
    zero_arr = np.zeros((1, raw_data.shape[1]))
    for i, (indices, flag) in enumerate(zip(mapping_result["index"], mapping_result["isOriginalData"])):
        if flag == True:
            new_data[i, :] = raw_data[indices[0], :]
        elif indices[0] == None:
            new_data[i, :] = zero_arr
        else:
            data = [raw_data[idx, :] for idx in indices]
            new_data[i, :] = np.mean(data, axis=0)
    return new_data

def preprocessing(filepath, inputfile, samplerate, mapping_result):
    # establish temp folder
    try:
        os.mkdir(filepath+"temp2/")
    except OSError as e:
        dataDelete(filepath+"temp2/")
        os.mkdir(filepath+"temp2/")
        print(e)
    
    # read data
    signal = read_train_data(inputfile)
    #print(signal.shape)
    # channel mapping
    signal = reorder_data(signal, mapping_result)
    #print(signal.shape)
    # resample
    signal = resample(signal, samplerate, 256)
    #print(signal.shape)
    # FIR_filter
    signal = FIR_filter(signal, 1, 50)
    #print(signal.shape)
    # cutting data
    total_file_num = cut_data(filepath, signal)

    return total_file_num

def restore_order(data, all_data, mapping_result):
    for i, (indices, flag) in enumerate(zip(mapping_result["index"], mapping_result["isOriginalData"])):
        if flag == True:
            all_data[indices[0], :] = data[i, :]
    return all_data

def postprocessing(data, samplerate, outputfile, mapping_result, batch_cnt, channel_num):
    
    # resample to original sampling rate
    data = resample(data, 256, samplerate)
    # reverse channel mapping
    all_data = np.zeros((channel_num, data.shape[1])) if batch_cnt==0 else read_train_data(outputfile)
    all_data = restore_order(data, all_data, mapping_result)
    # save data
    save_data(all_data, outputfile)


# model = tf.keras.models.load_model('./denoise_model/')
def reconstruct(model_name, total, filepath, batch_cnt):
    # -------------------decode_data---------------------------
    second1 = time.time()
    for i in range(total):
        file_name = filepath + 'temp2/{}.csv'.format(str(i))
        data_noise = read_train_data(file_name)
        
        std = np.std(data_noise)
        avg = np.average(data_noise)

        data_noise = (data_noise-avg)/std

        # Deep Learning Artifact Removal
        d_data = decode_data(data_noise, std, model_name)
        d_data = d_data[0]

        outputname = filepath + 'temp2/output{}.csv'.format(str(i))
        save_data(d_data, outputname)

    # --------------------glue_data----------------------------
    data = glue_data(filepath+"temp2/", total)
    # -------------------delete_data---------------------------
    dataDelete(filepath+"temp2/")
    second2 = time.time()
    
    print(f"Using {model_name} model to reconstruct batch-{batch_cnt+1} has been success in {second2 - second1} sec(s)")
    return data
    

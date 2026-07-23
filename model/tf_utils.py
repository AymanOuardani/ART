import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns


def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def save_checkpoint(state, save_path):
    """
    Save model checkpoint.
    :param state: model state
    :param is_best: is this checkpoint the best so far?
    :param save_path: the path for saving
    """
    filename = 'checkpoint.pth.tar'
    torch.save(state, os.path.join(save_path, filename))

def plotSignal(mode, signal, titlename):
    folderpath = "./testFig/"
    #titlename = "./testFig/Channel plot with mode" + str(mode)

    #print("draw:", mode, signal.shape)

    # Selecting one specific channel (e.g., the first channel)
    channel_to_plot = signal[0, :]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot the selected channel
    plt.plot(channel_to_plot)

    # Add labels and title
    plt.xlabel('Time or Sample Index')
    plt.ylabel('Amplitude')
    plt.title(titlename)

    # Save the figure to a file
    plt.savefig(folderpath + titlename+'.png')

def plotHeatmap(mode, data, titlename):
    folderpath = "./testFig/"
    #titlename = "Channel plot with mode" + str(mode)

    # print("draw:", mode, signal.shape)

    # Selecting one specific channel (e.g., the first channel)
    #data_2d = data[0, :, :]

    # Create a new figure
    plt.figure(figsize=(10, 8))

    # Plot the selected channel
    sns.heatmap(data, cmap="YlGnBu", cbar_kws={"shrink": 0.75})

    # Add labels and title
    plt.xlabel('Time point index')
    plt.ylabel('Time point index')
    plt.title(titlename)

    # Save the figure to a file
    plt.savefig(folderpath + titlename + '.png')

def draw(mode, signal, titlename):
    if mode == 0: #
        #signal = signal[0, :, :]
        signal = signal.cpu().detach().numpy()
        plotSignal(mode, signal, titlename)
    elif mode == 1:
        signal = signal[0, :, :]
        signal = signal.cpu().detach().numpy()
        plotSignal(mode, signal, titlename)
    elif mode == 2:
        #signal = signal[0, :, :]
        signal = torch.transpose(signal, 0, 1)
        signal = signal.cpu().detach().numpy()
        plotSignal(mode, signal, titlename)
    elif mode == 3: # plot headmap
        #signal = signal[0, :, :]
        #signal = torch.transpose(signal, 0, 1)
        signal = signal.cpu().detach().numpy()
        plotHeatmap(mode, signal, titlename)




import utils
import os

if __name__ == '__main__':
    # parameter setting
    input_path = './sampledata/'
    input_name = 'sampledata.csv'
    sample_rate = 256 # input data sample rate
    modelname = 'ICUNet' # or 'ICUNet', 'ICUNet++', 'ICUNet_attn', 'ART'
    output_path = './sampledata/'
    output_name = 'outputsample.csv'

    # read the mapping result
    mapping_name = './sampledata/sample_chanlocs_mapping_result.json'
    mapping_result, num_channel, num_group = utils.read_mapping_result(mapping_name)

    for i in range(num_group):

        # step1: Data preprocessing
        preprocess_data = utils.preprocessing(input_path+input_name, sample_rate, mapping_result[i])
        # step2: Signal reconstruction
        reconstructed_data = utils.reconstruct(modelname, preprocess_data, output_name, i)
        # step3: Data postprocessing
        utils.postprocessing(reconstructed_data, sample_rate, output_path+output_name, mapping_result[i], i, num_channel)

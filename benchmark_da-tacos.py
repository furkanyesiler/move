import argparse
import json
import os
import sys
from zipfile import ZipFile

import deepdish as dd
import gdown
import numpy as np
import torch


def download_da_tacos_cremapcp(output_dir, unpack_zips, remove_zips):
    """
    Function for downloading cremaPCP features of the Da-TACOS benchmark subset
    :param output_dir: output directory to store the features
    :param unpack_zips: whether to unpack the downloaded zip files
    :param remove_zips: whether to remove the downloaded zip files
    """

    if not os.path.exists(output_dir):  # checking whether the output directory exists
        sys.stderr.write('Output directory \'{}\' does not exist'.format(output_dir))
        return

    gdrive_prefix = 'https://drive.google.com/uc?id='  # prefix for downloading files from Google Drive

    crema_id = '13KJzwitploa2Teq4ndNcxcy2A2QOscpU'  # Google Drive ID of the cremaPCP zip file
    crema_filename = 'da-tacos_benchmark_subset_crema.zip'  # File name for the downloaded zip file

    metadata_id = '1y5g1whcd7bG6CiU0_cSqmpNR33hdKrby'  # Google Drive ID of the metadata zip file
    metadata_filename = 'da-tacos_metadata.zip'  # File name for the downloaded zip file

    print('Downloading metadata of Da-TACOS.')

    # downloading the metadata of Da-TACOS
    output = os.path.join(output_dir, metadata_filename)
    gdown.download('{}{}'.format(gdrive_prefix, metadata_id),
                   output,
                   quiet=False)

    print('Metadata is downloaded.')

    if unpack_zips:
        unpack_zip(output, output_dir)
    if remove_zips:
        remove_zip(output)

    print('Downloading the cremaPCP features of the Da-TACOS benchmark subset.')

    # downloading the cremaPCP features of the Da-TACOS benchmark subset
    output = os.path.join(output_dir, crema_filename)
    gdown.download('{}{}'.format(gdrive_prefix, crema_id),
                   output,
                   quiet=False)

    print('cremaPCP features are downloaded.')

    if unpack_zips:
        unpack_zip(output, output_dir)
    if remove_zips:
        remove_zip(output)


def unpack_zip(output, output_dir):
    """
    Function for unpacking the zip files
    :param output: the path of the zip file
    :param output_dir: directory to unpack
    """
    print('Unpacking the zip file {} into {}'.format(output, output_dir))
    with ZipFile(output, 'r') as z:
        z.extractall(output_dir)


def remove_zip(output):
    """
    Function for removing the zip files
    :param output: the path of the zip file
    """
    print('Removing the zip file {}'.format(output))
    os.remove(output)


def create_benchmark_pt(output_dir):
    """
    Function for preprocessing the cremaPCP features of the Da-TACOS benchmark subset.
    Preprocessed files are stored in output_dir directory as a single .pt file
    :param output_dir: the directory to store the .pt file
    :return: labels of the processed cremaPCP features
    """
    print('Creating benchmark_crema.pt file.')
    # reading the metadata file for the Da-TACOS benchmark subset
    with open(os.path.join(output_dir, 'da-tacos_metadata/da-tacos_benchmark_subset_metadata.json')) as f:
        metadata = json.load(f)

    data = []
    labels = []

    # iterating through the metadata file to create .pt file
    for key1 in metadata.keys():  # key1 specifies the work id
        for key2 in metadata[key1].keys():  # key2 specifies the performance id
            # loading the file
            temp_path = os.path.join(output_dir, 'da-tacos_benchmark_subset_crema/{}_crema/{}_crema.h5'.format(key1,
                                                                                                               key2))
            # reading cremaPCP features
            temp_crema = dd.io.load(temp_path)['crema']

            # downsampling the feature matrix and casting it as a torch.Tensor
            idxs = np.arange(0, temp_crema.shape[0], 8)
            temp_tensor = torch.from_numpy(temp_crema[idxs].T)

            # expanding in the pitch dimension, and adding the feature tensor and its label to the respective lists
            data.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
            labels.append(key1)

    # creating the dictionary to store
    benchmark_crema = {'data': data, 'labels': labels}

    # saving the .pt file
    torch.save(benchmark_crema, os.path.join(output_dir, 'benchmark_crema.pt'))

    return labels


def create_benchmark_ytrue(labels, output_dir):
    """
    Function for creating the ground truth file for evaluating models on the Da-TACOS benchmark subset.
    The created ground truth matrix is stored as a .pt file in output_dir directory
    :param labels: labels of the files
    :param output_dir: where to store the ground truth .pt file
    """
    print('Creating ytrue_benchmark.pt file.')
    ytrue = []  # empty list to store ground truth annotations
    for i in range(len(labels)):
        main_label = labels[i]  # label of the ith track in the list
        sub_ytrue = []  # empty list to store ground truth annotations for the ith track in the list
        for j in range(len(labels)):
            if labels[j] == main_label and i != j:  # checking whether the songs have the same label as the ith track
                sub_ytrue.append(1)
            else:
                sub_ytrue.append(0)
        ytrue.append(sub_ytrue)

    # saving the ground truth annotations
    torch.save(torch.Tensor(ytrue), os.path.join(output_dir, 'ytrue_benchmark.pt'))


def main(output_dir, unpack_zips, remove_zips):
    """
    Main function to download, preprocess and store the cremaPCP features of Da-TACOS benchmark subset and
    the necessary ground truth file for evaluation
    :param output_dir: output directory to store the features
    :param unpack_zips: whether to unpack the downloaded zip files
    :param remove_zips: whether to remove the downloaded zip files
    """
    download_da_tacos_cremapcp(output_dir=output_dir, unpack_zips=unpack_zips, remove_zips=remove_zips)

    if unpack_zips:
        labels = create_benchmark_pt(output_dir=output_dir)

        create_benchmark_ytrue(labels=labels, output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloading and preprocessing cremaPCP features of the '
                                                 'Da-TACOS benchmark subset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--outputdir',
                        default='./data',
                        help='Directory to store the dataset')
    parser.add_argument('--unpack', action='store_true', help='Unpack the zip files')
    parser.add_argument('--remove', action='store_true', help='Remove zip files after unpacking')

    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        raise Exception('The specified directory for storing the dataset does not exist.')

    main(args.outputdir, args.unpack, args.remove)
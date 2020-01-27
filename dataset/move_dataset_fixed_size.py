import numpy as np
import torch
from torch.utils.data import Dataset

from utils.move_utils import cs_augment


class MOVEDatasetFixed(Dataset):
    """
    MOVEDataset object returns 4 songs for a given label.
    Given features are pre-processed to have a same particular shape.
    """

    def __init__(self, data, labels, h=23, w=1800, data_aug=1, ytc=0):
        """
        Initialization function for the MOVEDataset object
        :param data: pcp features
        :param labels: labels of features (should be in the same order as features)
        :param h: height of pcp features (number of bins, e.g. 12 or 23)
        :param w: width of pcp features (number of frames in the temporal dimension)
        :param data_aug: whether to apply data augmentation to each song (1 or 0)=
        :param ytc: whether to train for benchmarking on YoutubeCovers dataset (1 or 0)
        """
        self.data = data  # pcp features
        self.labels = np.array(labels)  # labels of the pcp features

        self.seed = 42  # random seed
        self.h = h  # height of pcp features
        self.w = w  # width of pcp features
        self.data_aug = data_aug  # whether to apply data augmentation to each song

        self.labels_set = set(self.labels)  # the set of labels

        # dictionary to store which indexes belong to which label
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        # labels to exclude for benchmarking on YoutubeCovers dataset
        ytc_labels = ['W_6731', 'W_88', 'W_5327', 'W_21166', 'W_112698', 'W_726', 'W_4273', 'W_39452', 'W_34283',
                      'W_6430', 'W_134194', 'W_4517', 'W_14557', 'W_9036', 'W_16818', 'W_6414', 'W_6315']

        self.clique_list = []  # list to store all cliques

        # adding some cliques multiple times depending on their size
        for label in self.label_to_indices.keys():
            if ytc == 1:
                if label in ytc_labels:
                    continue
            if self.label_to_indices[label].size < 2:
                pass
            elif self.label_to_indices[label].size < 6:
                self.clique_list.extend([label] * 1)
            elif self.label_to_indices[label].size < 10:
                self.clique_list.extend([label] * 2)
            elif self.label_to_indices[label].size < 14:
                self.clique_list.extend([label] * 3)
            else:
                self.clique_list.extend([label] * 4)

    def __getitem__(self, index):
        """
        getitem function for the MOVEDataset object
        :param index: index of the clique picked by the dataloader
        :return: 4 songs and their labels from the picked clique
        """

        label = self.clique_list[index]  # getting the clique chosen by the dataloader

        # selecting 4 songs from the given clique
        if self.label_to_indices[label].size == 2:  # if the clique size is 2, repeat the already selected songs
            idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
            item1, item2 = self.data[idx1], self.data[idx2]
            item3, item4 = self.data[idx1], self.data[idx2]
        elif self.label_to_indices[label].size == 3:  # if the clique size is 3, choose one of the songs twice
            idx1, idx2, idx3 = np.random.choice(self.label_to_indices[label], 3, replace=False)
            idx4 = np.random.choice(self.label_to_indices[label], 1, replace=False)[0]
            item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
        else:  # if the clique size is larger than or equal to 4, choose 4 songs randomly
            idx1, idx2, idx3, idx4 = np.random.choice(self.label_to_indices[label], 4, replace=False)
            item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
        items_i = [item1, item2, item3, item4]  # list for storing selected songs

        items = []

        # pre-processing each song separately
        for item in items_i:
            if self.data_aug == 1:  # applying data augmentation to the song
                item = cs_augment(item)
            # if the song is longer than the required width, choose a random start point to crop
            if item.shape[2] >= self.w:
                p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                if len(p_index) != 0:
                    start = np.random.choice(p_index)
                    temp_item = item[:, :, start:start + self.w]
                    items.append(temp_item)
            else:  # if the song is shorter than the required width, zero-pad the end
                items.append(torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2))

        return torch.stack(items, 0), label

    def __len__(self):
        """
        Size of the MOVEDataset object
        :return: length of the clique list containing all the cliques (multiple cliques included for larger ones)
        """
        return len(self.clique_list)

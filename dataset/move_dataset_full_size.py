import numpy as np
from torch.utils.data import Dataset


class MOVEDatasetFull(Dataset):
    """
    MOVEDataset object returns one song from the test data.
    Given features are in their full length.
    """

    def __init__(self, data, labels):
        """
        Initialization of the MOVEDataset object
        :param data: pcp features
        :param labels: labels of features (should be in the same order as features)
        :param h: height of pcp features (number of bins, e.g. 12 or 23)
        :param w: width of pcp features (number of frames in the temporal dimension)
        """
        self.data = data  # pcp features
        self.labels = np.array(labels)  # labels of the pcp features

        self.labels_set = list(self.labels)  # the set of labels

        # dictionary to store which indexes belong to which label
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

    def __getitem__(self, index):
        """
        getitem function for the MOVEDataset object
        :param index: index of the song picked by the dataloader
        :return: pcp feature of the selected song
        """

        item = self.data[index]

        return item.float()

    def __len__(self):
        """
        Size of the MOVEDataset object
        :return: length of the entire dataset
        """
        return len(self.data)

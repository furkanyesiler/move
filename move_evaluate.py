import torch
from torch.utils.data import DataLoader

from dataset.move_dataset_full_size import MOVEDatasetFull
from models.move_model import MOVEModel
from models.move_model_nt import MOVEModelNT
from utils.move_utils import average_precision
from utils.move_utils import pairwise_distance_matrix
from utils.move_utils import import_dataset_from_pt


def test(move_model, test_loader, norm_dist=1):
    """
    Obtaining pairwise distances of all elements in the test set. For using full length songs,
    we pass them to the model one by one
    :param move_model: model to be used for testing
    :param test_loader: dataloader for test
    :param norm_dist: whether to normalize distances by the embedding size
    :return: pairwise distance matrix of all elements in the test set
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    with torch.no_grad():  # deactivating gradient tracking for testing
        move_model.eval()  # setting the model to evaluation mode

        # tensor for storing all the embeddings obtained from the test set
        embed_all = torch.tensor([], device=device)

        for batch_idx, item in enumerate(test_loader):

            if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
                item = item.cuda()

            res_1 = move_model(item)  # obtaining the embeddings of each song in the mini-batch

            embed_all = torch.cat((embed_all, res_1))  # adding the embedding of the current song to the others

        dist_all = pairwise_distance_matrix(embed_all)  # calculating the condensed distance matrix

        if norm_dist == 1:  # normalizing the distances
            dist_all /= move_model.fin_emb_size

    return dist_all


def evaluate(save_name,
             model_type,
             emb_size,
             sum_method,
             final_activation,
             dataset,
             dataset_name
             ):
    """
    Main evaluation function of MOVE. For a detailed explanation of parameters,
    please check 'python move_main.py -- help'
    :param save_name: name to save model and experiment summary
    :param model_type: which model to use: MOVE (0) or MOVE without transposition invariance (1)
    :param emb_size: the size of the final embeddings produced by the model
    :param sum_method: the summarization method for the model
    :param final_activation: final activation to use for the model
    :param dataset: which dataset to evaluate the model on. (0) validation set, (1) da-tacos, (2) ytc
    :param dataset_name: name of the file to evaluate
    """

    # indicating which dataset to use for evaluation
    # val_subset_crema is the name of our validation set
    if dataset_name == '':
        if dataset == 0:
            dataset_name = 'data/val_subset_crema.pt'
        elif dataset == 1:
            dataset_name = 'data/benchmark_crema.pt'
        else:
            dataset_name = 'data/ytc_crema.h5'
    else:
        dataset_name = 'data/{}'.format(dataset_name)

    print('Evaluating model {} on dataset {}.'.format(save_name, dataset_name))

    # initializing the model
    if model_type == 0:
        move_model = MOVEModel(emb_size=emb_size, sum_method=sum_method, final_activation=final_activation)
    elif model_type == 1:
        move_model = MOVEModelNT(emb_size=emb_size, sum_method=sum_method, final_activation=final_activation)

    # loading a pre-trained model
    model_name = 'saved_models/model_{}.pt'.format(save_name)

    move_model.load_state_dict(torch.load(model_name, map_location='cpu'))
    move_model.eval()

    # sending the model to gpu, if available
    if torch.cuda.is_available():
        move_model.cuda()

    # loading test data, initializing the dataset object and the data loader
    test_data, test_labels = import_dataset_from_pt(filename=dataset_name)
    test_map_set = MOVEDatasetFull(data=test_data, labels=test_labels)
    test_map_loader = DataLoader(test_map_set, batch_size=1, shuffle=False)

    # calculating the pairwise distances
    dist_map_matrix = test(move_model=move_model,
                           test_loader=test_map_loader).cpu()

    # calculating the performance metrics
    average_precision(
        -1 * dist_map_matrix.clone() + torch.diag(torch.ones(len(test_data)) * float('-inf')), dataset=dataset)

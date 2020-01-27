import torch
import torch.nn.functional as F

from utils.move_utils import pairwise_distance_matrix


def triplet_loss_mining(res_1, move_model, labels, margin=1, mining_strategy=2, norm_dist=1):
    """
    Online mining function for selecting the triplets
    :param res_1: embeddings in the mini-batch
    :param move_model: model used to obtain the embeddings
    :param margin: margin for the triplet loss
    :param mining_strategy: which mining strategy to use (0 for random, 1 for semi-hard, 2 for hard)
    :param norm_dist: whether to normalize the distances by the embedding size
    :param labels: labels of the embeddings
    :return: triplet loss value
    """

    # creating positive and negative masks for online mining
    aux = {}
    i_labels = []
    for l in labels:
        if l not in aux:
            aux[l] = len(aux)
        i_labels += [aux[l]]*4

    i_labels = torch.Tensor(i_labels).view(-1, 1)
    mask_diag = (1 - torch.eye(res_1.size(0))).long()
    if torch.cuda.is_available():
        i_labels = i_labels.cuda()
        mask_diag = mask_diag.cuda()
    temp_mask = (pairwise_distance_matrix(i_labels) < 0.5).long()
    mask_pos = mask_diag * temp_mask
    mask_neg = mask_diag * (1 - mask_pos)

    dist_all = pairwise_distance_matrix(res_1)  # getting the pairwise distance matrix
    if norm_dist == 1:  # normalizing the distances by the embedding size
        dist_all /= move_model.fin_emb_size

    if mining_strategy == 0:  # random mining
        dist_g, dist_i = triplet_mining_random(dist_all, mask_pos, mask_neg)
    elif mining_strategy == 1:  # semi-hard mining
        dist_g, dist_i = triplet_mining_semihard(dist_all, mask_pos, mask_neg)
    else:  # hard mining
        dist_g, dist_i = triplet_mining_hard(dist_all, mask_pos, mask_neg)

    loss = F.relu(dist_g + (margin - dist_i))  # calculating triplet loss

    return loss.mean()


def triplet_mining_hard(dist_all, mask_pos, mask_neg):
    """
    Performs online hard triplet mining (both positive and negative)
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # selecting the positive elements of triplets
    _, sel_pos = torch.max(dist_all * mask_pos.float(), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # modifying the negative mask for hard mining
    mask_neg = torch.where(mask_neg == 0, torch.tensor(float('inf'), device=device), torch.tensor(1., device=device))

    # selecting the negative elements of triplets
    _, sel_neg = torch.min(dist_all + mask_neg.float(), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_random(dist_all, mask_pos, mask_neg):
    """
    Performs online random triplet mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # selecting the negative elements of triplets
    _, sel_neg = torch.max(mask_neg.float() + torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_semihard(dist_all, mask_pos, mask_neg):
    """
    Performs online semi-hard triplet mining (a random positive, a semi-hard negative)
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # selecting the negative elements of triplets
    _, sel_neg = torch.max((mask_neg + mask_neg * (dist_all < dists_pos.expand_as(dist_all)).long()).float() +
                           torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


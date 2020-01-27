import argparse
import json

from move_evaluate import evaluate
from move_train import train

if __name__:
    with open('data/move_defaults.json') as f:
        defaults = json.load(f)

    parser = argparse.ArgumentParser(description='Training code of MOVE')
    parser.add_argument('-rt',
                        '--run_type',
                        type=str,
                        default='train',
                        choices=('train', 'test'),
                        help='Whether to run train or test script')
    parser.add_argument('-tp',
                        '--train_path',
                        type=str,
                        default=defaults['train_path'],
                        help='Path for training data. If more than one file are used, '
                             'write only the common part')
    parser.add_argument('-ch',
                        '--chunks',
                        type=int,
                        default=defaults['chunks'],
                        help='Number of chunks for training set')
    parser.add_argument('-vp',
                        '--val_path',
                        type=str,
                        default=defaults['val_path'],
                        help='Path for validation data')
    parser.add_argument('-sm',
                        '--save_model',
                        type=int,
                        default=defaults['save_model'],
                        choices=(0, 1),
                        help='1 for saving the trained model, 0 for otherwise')
    parser.add_argument('-ss',
                        '--save_summary',
                        type=int,
                        default=defaults['save_summary'],
                        choices=(0, 1),
                        help='1 for saving the training log, 0 for otherwise')
    parser.add_argument('-rs',
                        '--random_seed',
                        type=int,
                        default=defaults['random_seed'],
                        help='Random seed')
    parser.add_argument('-noe',
                        '--num_of_epochs',
                        type=int,
                        default=defaults['num_of_epochs'],
                        help='Number of epochs for training')
    parser.add_argument('-m',
                        '--model_type',
                        type=int,
                        default=defaults['model_type'],
                        choices=(0, 1),
                        help='0 for MOVE, 1 for MOVE without pitch transposition')
    parser.add_argument('-emb',
                        '--emb_size',
                        type=int,
                        default=defaults['emb_size'],
                        help='Size of the final embeddings')
    parser.add_argument('-sum',
                        '--sum_method',
                        type=int,
                        choices=(0, 1, 2, 3, 4),
                        default=defaults['sum_method'],
                        help='0 for max-pool, 1 for mean-pool, 2 for autopool, '
                             '3 for multi-channel attention, 4 for multi-channel adaptive attention')
    parser.add_argument('-fa',
                        '--final_activation',
                        type=int,
                        choices=(0, 1, 2, 3),
                        default=defaults['final_activation'],
                        help='0 for no activation, 1 for sigmoid, 2 for tanh, 3 for batch norm')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=defaults['learning_rate'],
                        help='Initial learning rate')
    parser.add_argument('-lrs',
                        '--lr_schedule',
                        type=int,
                        default=defaults['lr_schedule'],
                        choices=(0, 1, 2),
                        help='0 for no lr_schedule, 1 for decreasing lr at epoch 80, '
                             '2 for decreasing lr at epochs [80, 100]')
    parser.add_argument('-lrsf',
                        '--lrsch_factor',
                        type=float,
                        default=defaults['lrsch_factor'],
                        help='Factor for lr scheduler')
    parser.add_argument('-mo',
                        '--momentum',
                        type=float,
                        default=defaults['momentum'],
                        help='Value for momentum parameter for SGD')
    parser.add_argument('-pl',
                        '--patch_len',
                        type=int,
                        default=defaults['patch_len'],
                        help='Size of the input len in time dimension')
    parser.add_argument('-nol',
                        '--num_of_labels',
                        type=int,
                        default=defaults['num_of_labels'],
                        help='Number of cliques per batch for triplet mining')
    parser.add_argument('-da',
                        '--data_aug',
                        type=int,
                        choices=(0, 1),
                        default=defaults['data_aug'],
                        help='0 for no data aug, 1 using it')
    parser.add_argument('-nd',
                        '--norm_dist',
                        type=int,
                        choices=(0, 1),
                        default=defaults['norm_dist'],
                        help='1 for normalizing the distance, 0 for avoiding it')
    parser.add_argument('-ms',
                        '--mining_strategy',
                        type=int,
                        default=defaults['mining_strategy'],
                        choices=(0, 1, 2),
                        help='0 for only random, 1 for only semi-hard, 2 for only hard')
    parser.add_argument('-ma',
                        '--margin',
                        type=float,
                        default=defaults['margin'],
                        help='Margin for triplet loss')
    parser.add_argument('-ytc',
                        '--ytc_labels',
                        type=int,
                        default=defaults['ytc_labels'],
                        choices=(0, 1),
                        help='0 for using full training data, 1 for removing overlapping labels with ytc')
    parser.add_argument('-d',
                        '--dataset',
                        type=int,
                        choices=(0, 1, 2),
                        default=0,
                        help='Choosing evaluation set for testing. 0 for move validation, '
                             '1 for test on da-tacos, 2 for test on ytc')
    parser.add_argument('-dn',
                        '--dataset_name',
                        type=str,
                        default='',
                        help='Specifying a dataset name for evaluation. '
                             'The dataset must be located in the data folder')

    args = parser.parse_args()

    lr_arg = '{}'.format(args.learning_rate).replace('.', '-')
    margin_arg = '{}'.format(args.margin).replace('.', '-')

    save_name = 'move'

    for key in defaults.keys():
        if key == 'abbr':
            pass
        else:
            if defaults[key] != getattr(args, key):
                save_name = '{}_{}_{}'.format(save_name, defaults['abbr'][key], getattr(args, key))

    if args.run_type == 'train':
        train(save_name=save_name,
              train_path=args.train_path,
              chunks=args.chunks,
              val_path=args.val_path,
              save_model=args.save_model,
              save_summary=args.save_summary,
              seed=args.random_seed,
              num_of_epochs=args.num_of_epochs,
              model_type=args.model_type,
              emb_size=args.emb_size,
              sum_method=args.sum_method,
              final_activation=args.final_activation,
              lr=args.learning_rate,
              lrsch=args.lr_schedule,
              lrsch_factor=args.lrsch_factor,
              momentum=args.momentum,
              patch_len=args.patch_len,
              num_of_labels=args.num_of_labels,
              ytc=args.ytc_labels,
              data_aug=args.data_aug,
              norm_dist=args.norm_dist,
              mining_strategy=args.mining_strategy,
              margin=args.margin
              )
    else:
        evaluate(save_name=save_name,
                 model_type=args.model_type,
                 emb_size=args.emb_size,
                 sum_method=args.sum_method,
                 final_activation=args.final_activation,
                 dataset=args.dataset,
                 dataset_name=args.dataset_name)

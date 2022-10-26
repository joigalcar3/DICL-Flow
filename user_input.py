import argparse


def user_input():
    parser = argparse.ArgumentParser(description='Parser for training and testing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data',help='path to dataset')
    parser.add_argument('--data-kitti12',help='path to kitti2012, if necessary', default=None)
    parser.add_argument('--cfg', dest='cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--split-file', default=None, type=str,help='test-val split file')
    parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                        help='solver algorithms')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                        help='manual epoch size (will match dataset size if set to 0)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--pretrained', dest='pretrained', default=None,
                        help='path to pre-trained model')

    parser.add_argument('--milestones', default=[300,500,1000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--eval_freq', default=1, type=int,
                        help='evaluation frequency')
    parser.add_argument('--no-eval', action='store_true',
                        help='do not conduct validation ')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--reuse_optim', action='store_true',
                        help='reuse optimizer ')
    parser.add_argument('--eval_sintel', action='store_true',
                        help='eval on sintel ')
    parser.add_argument('--out_dir', default=None,
                        help='path to save model')
    parser.add_argument('--exp_dir', default='default',
                        help='path to save model')
    parser.add_argument('--visual_all', action='store_true',
                        help='eval on sintel ')
    parser.add_argument('--clip', default=-1, type=float)
    return parser, group
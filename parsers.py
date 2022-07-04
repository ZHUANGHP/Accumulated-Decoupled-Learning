import argparse
from torch.backends import cudnn
import torch

parser = argparse.ArgumentParser(description='PyTorch decoupled training')
parser.add_argument('--type', default='ADL',
                    help='training with FDG, DDG, FR or DGL (default: FDG)')
parser.add_argument('--dir', type=str, default='/home/zhuang/dataset/ImageNet',
                    help='dataset dir')
parser.add_argument('--mode', default='B',
                    help='mode A or mode B (default B)')
parser.add_argument('--model', default='ResNet20',
                    help='models, ResNet18, ResNet50, ResNet101, ResNet20, ResNet56, ResNet110, ResNet164, ResNet1202, WRN28_10 (default: ResNet18)')
parser.add_argument('--backprop', action='store_true', default=False,
                    help='backprop')
parser.add_argument('--num-split', type=int, default=3,
                    help='the number of splits of the model (default: 2)')
parser.add_argument('--dataset', default='CIFAR10',
                    help='dataset, MNIST, KuzushijiMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10 or ImageNet (default: CIFAR10)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate (default: 1e-1)')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[150,225,275],
                    help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
parser.add_argument('--lr-decay-fact', type=float, default=0.1,
                    help='learning rate decay factor to use at milestone epochs (default: 0.25)')
parser.add_argument('--warm-up-epochs', type=int, default=5,
                    help='warming up epochs')
parser.add_argument('--optim', default='SGD',
                    help='optimizer, adam, amsgrad or sgd (default: SGD)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='weight decay (default: 0.0)')
parser.add_argument('--beta', type=float, default=1,
                    help='shrinking the learning rate in earlier modules (default: 1)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout after each nonlinearity (default: 0.1)')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='enable warming up for 3 epochs (default: False)')
parser.add_argument('--pbar', action='store_true', default=False,
                    help='show progress bar during training')
parser.add_argument('--save', action='store_true', default=False,
                    help='save the model (default: False)')
parser.add_argument('--comment', type=str, default=None,
                    help='comments')
parser.add_argument('--writer', action='store_true', default=False,
                    help='enable writer')
parser.add_argument('--workers', type=int, default=0,
                    help='')
parser.add_argument('--ac-step', type=int, default=1,
                    help='')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='')
parser.add_argument('--prof', action='store_true', default=False,
                    help='')
parser.add_argument('--mulgpu', action='store_true', default=False,
                    help='use multiple gpus')
parser.add_argument('--datapal', action='store_true', default=False,
                    help='data parallelization')

args = parser.parse_args()
#args.backprop = True

def get_string(args):
    if args.backprop:
        string_print = 'BP' + '_' + args.model + '_' + args.dataset + '_' + '_bs' + str(args.batch_size) + '_lr' + str(args.lr)+\
                       ('_mulgpu' if args.mulgpu is True else '') + ('_warm' if args.warm_up is True else '') + \
                       ('_s' + str(args.seed) if args.seed is not None else '') + ('_cmmnt_' + str(args.comment) if args.comment is not None else '')
    else:
        string_print = str(args.type) + '_' + args.mode + '_Split' + str(args.num_split) + '_' + args.model + '_' + args.dataset +\
                       '_lr' + str(args.lr) + 'dcy' + str(args.lr_decay_fact) + '_beta' + str(args.beta) + '_bs' + str(args.batch_size)+\
                       '_ac' + str(args.ac_step) + \
                       ('_mulgpu' if args.mulgpu is True else '') + ('_warm' if args.warm_up is True else '') + ('_datapal' if args.datapal is True else '') + \
                       ('_s'+str(args.seed) if args.seed is not None else '') + ('_cmmnt_' + str(args.comment) if args.comment is not None else '')
    return string_print


string_print = get_string(args)
print(string_print)
print(args)
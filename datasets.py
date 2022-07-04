import numpy as np
import torch
from torchvision import datasets, transforms
from parsers import args
import random
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]


kwargs = {'num_workers': args.workers, 'pin_memory': True}
if args.dataset == 'MNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data/MNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'FashionMNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,))
    ])
    dataset_train = datasets.FashionMNIST('../data/FashionMNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/FashionMNIST', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.286,), (0.353,))
                              ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'KuzushijiMNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1904,), (0.3475,))
    ])
    dataset_train = KuzushijiMNIST('../data/KuzushijiMNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        KuzushijiMNIST('../data/KuzushijiMNIST', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1904,), (0.3475,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'CIFAR10':
    input_dim = 32
    input_ch = 3
    num_classes = 10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    dataset_train = datasets.CIFAR10(root='../data/CIFAR10/', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data/CIFAR10/', train=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                         ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'CIFAR100':
    input_dim = 32
    input_ch = 3
    num_classes = 100
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    dataset_train = datasets.CIFAR100('../data/CIFAR100', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data/CIFAR100', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                          ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'SVHN':
    input_dim = 32
    input_ch = 3
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
    ])
    dataset_train = torch.utils.data.ConcatDataset((
        datasets.SVHN('../data/SVHN', split='train', download=True, transform=train_transform),
        datasets.SVHN('../data/SVHN', split='extra', download=True, transform=train_transform)))
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data/SVHN', split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
                      ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'STL10':
    input_dim = 96
    input_ch = 3
    num_classes = 10
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
    ])
    dataset_train = datasets.STL10('../data/STL10', split='train', download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='test',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'tiny-imagenet':
    input_dim = 64
    input_ch = 3
    num_classes = 200
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])
    dataset_train = datasets.ImageFolder(args.dir+'/train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs
                                               )
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.dir+'/val',
                             transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                             ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'imagenet':
    input_dim = 224
    input_ch = 3
    num_classes = 1000
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder(args.dir+'/train', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.dir+'/val2',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
else:
    print('No valid dataset is specified')

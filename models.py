from parsers import args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LRS
import torch.optim as optim
from collections import deque
import os
import sys
from datasets import input_dim, input_ch, num_classes
from ResNet_ImageNet import Flatten
import math
class BuildDG(nn.Module):
    def __init__(self, model, optimizer, scheduler, split_loc):
        super(BuildDG, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.type == 'FDG' or args.type == 'ADL' or args.type == 'DGL':
            self.delay = 2*(args.num_split - split_loc - 1)
            output_dq = self.delay
        elif args.type == 'FR' or args.type == 'DDG':
            self.delay = (args.num_split - split_loc - 1)
            output_dq = self.delay + 1
        else:
            print('No learning method has been specified!')
        self.dg = None
        self.update_count = 0
        self.output = deque(maxlen=output_dq)
        self.module_num = split_loc
        for x in range(output_dq):
            self.output.append(None)
        self.input = deque(maxlen=self.delay+1)
        for x in range(self.delay+1):
            self.input.append(None)
        self.input_grad = None
        self.acc = 0
        self.acc5 = 0
        self.loss = 0
        if split_loc == 0:
            self.first_layer = True
            self.last_layer = False
        elif split_loc == args.num_split-1:
            self.first_layer = False
            self.last_layer = True
        else:
            self.first_layer = False
            self.last_layer = False
        
    def forward(self, x):
        out = self.model(x)
        return out

    def backward(self):
        graph = self.output.popleft()
        if self.dg is not None and graph is not None:
            rev_grad = True
            graph.backward(args.beta*self.dg)
            self.update_count += 1
        else:
            rev_grad = False
            print('no backward in module {}'.format(self.module_num))

        return rev_grad
    
    def get_grad(self):
        return self.input.popleft().grad
    
    def get_output(self):
        return self.output[self.delay-1]
    
    def train(self):
        self.model.train()
        
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()


device = {}
if torch.cuda.is_available():
    if args.mulgpu:
        for i in range(args.num_split):
            device[i] = torch.device('cuda:'+str(i))
    else:
        for i in range(args.num_split):
            device[i] = torch.device('cuda:'+str(0))
    '''
    if args.backprop:
        for i in range(args.num_split):
            device[i] = torch.device('cuda:' + str(0))
    '''
else:
    for i in range(args.num_split):
        device[i] = torch.device('cpu')


class BuildAuxNet(nn.Module):
    def __init__(self, model, optimizer, scheduler, split_loc):
        super(BuildAuxNet, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.module_num = split_loc

    def forward(self, x):
        out = self.model(x)
        return out

    def backward(self, loss):
        loss.backward()

    def train(self):
        self.model.train()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class auxillary_dgl(nn.Module):
    def __init__(self, feature_size=256,
                 input_features=256, in_size=32,
                 num_classes=10, n_lin=3, mlp_layers=3, batchn=True):
        super(auxillary_dgl, self).__init__()
        self.n_lin = n_lin
        self.in_size = in_size

        if n_lin == 0:
            feature_size = input_features

        feature_size = input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n == 0:
                input_features = input_features
            else:
                input_features = feature_size

            if batchn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = identity()

            conv = nn.Conv2d(input_features, feature_size,
                             kernel_size=1, stride=1, padding=0, bias=False)
            self.blocks.append(nn.Sequential(conv, bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        if batchn:
            self.bn = nn.BatchNorm2d(feature_size)
        else:
            self.bn = identity()  # Identity

        if mlp_layers > 0:

            mlp_feat = feature_size * (2) * (2)
            layers = []

            for l in range(mlp_layers):
                if l == 0:
                    in_feat = feature_size * 4
                    mlp_feat = mlp_feat
                else:
                    in_feat = mlp_feat

                if batchn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = identity()

                layers += [nn.Linear(in_feat, mlp_feat),
                           bn_temp, nn.ReLU(True)]
            layers += [nn.Linear(mlp_feat, num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True

        else:
            self.mlp = False
            self.classifier = nn.Linear(feature_size * (in_size // avg_size) * (in_size // avg_size), num_classes)

    def forward(self, x):
        out = x
        # First reduce the size by 16
        out = F.adaptive_avg_pool2d(out, (math.ceil(self.in_size / 4), math.ceil(self.in_size / 4)))

        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, (2, 2))
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


model_list = {}
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        optimizer_save = checkpoint['optimizer']
        resume_dir = args.resume
        args = checkpoint['args']
        args.resume = resume_dir
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    args.start_epoch = 0
    best_prec1 = 0

if args.model == 'alexnet':
    import torchvision.models as models
    model = models.alexnet()
    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.features[:3])
        model_list[1] = nn.Sequential(model.features[3:6])
        model_list[2] = nn.Sequential(model.features[6:8])
        model_list[3] = nn.Sequential(model.features[8:10])
        model_list[4] = nn.Sequential(model.features[10:])
        model_list[5] = nn.Sequential(model.avgpool, Flatten(), model.classifier[:3])
        model_list[6] = nn.Sequential(model.classifier[3:6])
        model_list[7] = nn.Sequential(model.classifier[6:])
elif args.model == 'ResNet18':
    if args.dataset == 'imagenet' or args.dataset == 'STL10':
        import torchvision.models as models
        model = models.resnet18()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet18(num_classes=num_classes)
        model.maxpool = nn.Sequential()
        if args.resume:
            model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 2:
        '''
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:0])
        model_list[1] = nn.Sequential(model.layer2[0:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        '''
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:1])
        model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[0:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        # model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten, model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, )
        model_list[1] = nn.Sequential(model.layer1, model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1)
        model_list[2] = nn.Sequential(model.layer2)
        model_list[3] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1], model.layer2[0])
        model_list[3] = nn.Sequential(model.layer2[1], model.layer3[0])
        model_list[4] = nn.Sequential(model.layer3[1], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 6:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1], model.layer2[0])
        model_list[3] = nn.Sequential(model.layer2[1])
        model_list[4] = nn.Sequential(model.layer3)
        model_list[5] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 7:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3)
        model_list[6] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3[0])
        model_list[6] = nn.Sequential(model.layer3[1])
        model_list[7] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 9:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3[0])
        model_list[6] = nn.Sequential(model.layer3[1])
        model_list[7] = nn.Sequential(model.layer4[0])
        model_list[8] = nn.Sequential(model.layer4[1], model.avgpool, Flatten(), model.fc)
    if args.num_split == 10:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3[0])
        model_list[6] = nn.Sequential(model.layer3[1])
        model_list[7] = nn.Sequential(model.layer4[0])
        model_list[8] = nn.Sequential(model.layer4[1], model.avgpool)
        model_list[9] = nn.Sequential(Flatten(), model.fc)
elif args.model == 'ResNet34':
    if args.dataset == 'imagenet' or args.dataset == 'STL10':
        import senet as models
        model = models.resnet34()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet34(num_classes=num_classes)
        model.maxpool = nn.Sequential()
        if args.resume:
            model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        #model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten, model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:1])
        model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[:1])
        model_list[2] = nn.Sequential(model.layer2[1:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1)
        model_list[2] = nn.Sequential(model.layer2[:3])
        model_list[3] = nn.Sequential(model.layer2[3:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:])
        model_list[3] = nn.Sequential(model.layer2)
        model_list[4] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 6:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2)
        model_list[5] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 7:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:])
        model_list[6] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:])
        model_list[6] = nn.Sequential(model.layer3[:2])
        model_list[7] = nn.Sequential(model.layer3[2:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 9:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:4])
        model_list[6] = nn.Sequential(model.layer3[:2])
        model_list[7] = nn.Sequential(model.layer3[2:4])
        model_list[8] = nn.Sequential(model.layer3[4:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 10:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:4])
        model_list[6] = nn.Sequential(model.layer3[:2])
        model_list[7] = nn.Sequential(model.layer3[2:4])
        model_list[8] = nn.Sequential(model.layer3[4:])
        model_list[9] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 11:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2:4])
        model_list[7] = nn.Sequential(model.layer3[:2])
        model_list[8] = nn.Sequential(model.layer3[2:4])
        model_list[9] = nn.Sequential(model.layer3[4:])
        model_list[10] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 12:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[:2])
        model_list[9] = nn.Sequential(model.layer3[2:4])
        model_list[10] = nn.Sequential(model.layer3[4:])
        model_list[11] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 13:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[0])
        model_list[9] = nn.Sequential(model.layer3[1])
        model_list[10] = nn.Sequential(model.layer3[2:4])
        model_list[11] = nn.Sequential(model.layer3[4:])
        model_list[12] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 14:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[0])
        model_list[9] = nn.Sequential(model.layer3[1])
        model_list[10] = nn.Sequential(model.layer3[2])
        model_list[11] = nn.Sequential(model.layer3[3:4])
        model_list[12] = nn.Sequential(model.layer3[4:])
        model_list[13] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 15:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[0])
        model_list[9] = nn.Sequential(model.layer3[1])
        model_list[10] = nn.Sequential(model.layer3[2])
        model_list[11] = nn.Sequential(model.layer3[3])
        model_list[12] = nn.Sequential(model.layer3[4])
        model_list[13] = nn.Sequential(model.layer3[5])
        model_list[14] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 16:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[0])
        model_list[9] = nn.Sequential(model.layer3[1])
        model_list[10] = nn.Sequential(model.layer3[2])
        model_list[11] = nn.Sequential(model.layer3[3])
        model_list[12] = nn.Sequential(model.layer3[4])
        model_list[13] = nn.Sequential(model.layer3[5])
        model_list[14] = nn.Sequential(model.layer4[0])
        model_list[15] = nn.Sequential(model.layer4[1:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 17:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer2[3])
        model_list[8] = nn.Sequential(model.layer3[0])
        model_list[9] = nn.Sequential(model.layer3[1])
        model_list[10] = nn.Sequential(model.layer3[2])
        model_list[11] = nn.Sequential(model.layer3[3])
        model_list[12] = nn.Sequential(model.layer3[4])
        model_list[13] = nn.Sequential(model.layer3[5])
        model_list[14] = nn.Sequential(model.layer4[0])
        model_list[15] = nn.Sequential(model.layer4[1])
        model_list[16] = nn.Sequential(model.layer4[2], model.avgpool, Flatten(), model.fc)
elif args.model == 'seResNet18':
    if args.dataset == 'imagenet' or args.dataset == 'STL10':
        import senet as models
        model = models.se_resnet18()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet18(num_classes=num_classes)
        model.maxpool = nn.Sequential()
        if args.resume:
            model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:0])
        model_list[1] = nn.Sequential(model.layer2[0:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        #model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten, model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:1])
        model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[:1])
        model_list[2] = nn.Sequential(model.layer2[1:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1)
        model_list[2] = nn.Sequential(model.layer2)
        model_list[3] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3[0])
        model_list[6] = nn.Sequential(model.layer3[1])
        model_list[7] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 10:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer2[0])
        model_list[4] = nn.Sequential(model.layer2[1])
        model_list[5] = nn.Sequential(model.layer3[0])
        model_list[6] = nn.Sequential(model.layer3[1])
        model_list[7] = nn.Sequential(model.layer4[0])
        model_list[8] = nn.Sequential(model.layer4[1], model.avgpool)
        model_list[9] = nn.Sequential(Flatten(), model.fc)
elif args.model == 'seResNet34':
    if args.dataset == 'imagenet' or args.dataset == 'STL10':
        import senet as models
        model = models.se_resnet34()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet34(num_classes=num_classes)
        model.maxpool = nn.Sequential()
        if args.resume:
            model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        #model_list[1] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten, model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:1])
        model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[:1])
        model_list[2] = nn.Sequential(model.layer2[1:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        model_list[1] = nn.Sequential(model.layer1)
        model_list[2] = nn.Sequential(model.layer2[:3])
        model_list[3] = nn.Sequential(model.layer2[3:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:])
        model_list[6] = nn.Sequential(model.layer3[:2])
        model_list[7] = nn.Sequential(model.layer3[2:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 10:
        model_list[0] = nn.Sequential(model.conv1)
        model_list[1] = nn.Sequential(model.bn1, model.relu, model.maxpool, model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1:2])
        model_list[3] = nn.Sequential(model.layer1[2:])
        model_list[4] = nn.Sequential(model.layer2[:2])
        model_list[5] = nn.Sequential(model.layer2[2:4])
        model_list[6] = nn.Sequential(model.layer3[:2])
        model_list[7] = nn.Sequential(model.layer3[2:4])
        model_list[8] = nn.Sequential(model.layer3[4:])
        model_list[9] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet50':

    if args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.resnet50()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet50(num_classes=num_classes)
        model.maxpool = nn.Sequential()
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:3])
        model_list[1] = nn.Sequential(model.layer2[3:], model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:2])
        model_list[1] = nn.Sequential(model.layer2[2:], model.layer3[:3])
        model_list[2] = nn.Sequential(model.layer3[3:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1,model.relu, model.maxpool, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:], model.layer2[:2])
        model_list[2] = nn.Sequential(model.layer2[2:], model.layer3[:2])
        model_list[3] = nn.Sequential(model.layer3[2:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.bn1,model.relu, model.maxpool, model.layer1[:1])
        model_list[1] = nn.Sequential(model.layer1[1:])
        model_list[2] = nn.Sequential(model.layer2[:2])
        model_list[3] = nn.Sequential(model.layer2[2:])
        model_list[4] = nn.Sequential(model.layer3[:2])
        model_list[5] = nn.Sequential(model.layer3[2:4])
        model_list[6] = nn.Sequential(model.layer3[4:])
        model_list[7] = nn.Sequential(model.layer4, model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet101':
    if args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.resnet101()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
    else:
        import ResNet_ImageNet as ResNet
        model = ResNet.ResNet101(num_classes=num_classes)
        model.maxpool = nn.Sequential()
        print('cifar10!')
        # bs = 64 (6.35*64) backprop = 64 (3.35*64)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3[:6])
        model_list[1] = nn.Sequential(model.layer3[6:], model.layer4, model.avgpool, Flatten(), model.fc)
        # bs = 128 (4.66*128), backprop = 64 (3.35*64. 2.21*100)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:4])
        model_list[1] = nn.Sequential(model.layer2[4:], model.layer3[:13])
        model_list[2] = nn.Sequential(model.layer3[13:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3[:5])
        model_list[2] = nn.Sequential(model.layer3[5:16])
        model_list[3] = nn.Sequential(model.layer3[16:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2[:3])
        model_list[1] = nn.Sequential(model.layer2[3:], model.layer3[:6])
        model_list[2] = nn.Sequential(model.layer3[6:13])
        model_list[3] = nn.Sequential(model.layer3[13:20])
        model_list[4] = nn.Sequential(model.layer3[20:], model.layer4, model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:], model.layer2[:1])
        model_list[2] = nn.Sequential(model.layer2[1:])
        model_list[3] = nn.Sequential(model.layer3[:5])
        model_list[4] = nn.Sequential(model.layer3[5:10])
        model_list[5] = nn.Sequential(model.layer3[10:16])
        model_list[6] = nn.Sequential(model.layer3[16:21])
        model_list[7] = nn.Sequential(model.layer3[21:], model.layer4, model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet20':
    import ResNet_cifar as ResNet
    model = ResNet.resnet20(num_classes=num_classes)
    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.avgpool, Flatten(), model.fc)

    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu)
        model_list[1] = nn.Sequential(model.layer1)
        model_list[2] = nn.Sequential(model.layer2)
        model_list[3] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 11:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu)
        model_list[1] = nn.Sequential(model.layer1[0])
        model_list[2] = nn.Sequential(model.layer1[1])
        model_list[3] = nn.Sequential(model.layer1[2])
        model_list[4] = nn.Sequential(model.layer2[0])
        model_list[5] = nn.Sequential(model.layer2[1])
        model_list[6] = nn.Sequential(model.layer2[2])
        model_list[7] = nn.Sequential(model.layer3[0])
        model_list[8] = nn.Sequential(model.layer3[1])
        model_list[9] = nn.Sequential(model.layer3[2], model.avgpool)
        model_list[10] = nn.Sequential(Flatten(), model.fc)
elif args.model == 'ResNet56':
    import ResNet_cifar as ResNet
    model = ResNet.resnet56(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:4])
        model_list[1] = nn.Sequential(model.layer2[4:], model.layer3, model.avgpool, Flatten(), model.fc)
    elif args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
    elif args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:6])
        model_list[1] = nn.Sequential(model.layer1[6:], model.layer2[:4])
        model_list[2] = nn.Sequential(model.layer2[4:], model.layer3[:2])
        model_list[3] = nn.Sequential(model.layer3[2:], model.avgpool, Flatten(), model.fc)
    elif args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:3])
        model_list[1] = nn.Sequential(model.layer1[3:])
        model_list[2] = nn.Sequential(model.layer2[:4])
        model_list[3] = nn.Sequential(model.layer2[4:], model.layer3[:2])
        model_list[4] = nn.Sequential(model.layer3[2:], model.avgpool, Flatten(), model.fc)
    elif args.num_split == 6:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:6])
        model_list[2] = nn.Sequential(model.layer1[6:9], model.layer2[:1])
        model_list[3] = nn.Sequential(model.layer2[1:9])
        model_list[4] = nn.Sequential(model.layer3[:8])
        model_list[5] = nn.Sequential(model.layer3[8:], model.avgpool, Flatten(), model.fc)
    elif args.num_split == 7:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:6])
        model_list[2] = nn.Sequential(model.layer1[6:9], model.layer2[:1])
        model_list[3] = nn.Sequential(model.layer2[1:5])
        model_list[4] = nn.Sequential(model.layer2[5:9])
        model_list[5] = nn.Sequential(model.layer3[:4])
        model_list[6] = nn.Sequential(model.layer3[4:], model.avgpool, Flatten(), model.fc)
    elif args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:6])
        model_list[2] = nn.Sequential(model.layer1[6:9], model.layer2[:1])
        model_list[3] = nn.Sequential(model.layer2[1:5])
        model_list[4] = nn.Sequential(model.layer2[5:9])
        model_list[5] = nn.Sequential(model.layer3[:4])
        model_list[6] = nn.Sequential(model.layer3[4:8])
        model_list[7] = nn.Sequential(model.layer3[8:], model.avgpool, Flatten(), model.fc)
    elif args.num_split == 16:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu)
        model_list[1] = nn.Sequential(model.layer1[:2])
        model_list[2] = nn.Sequential(model.layer1[2:4])
        model_list[3] = nn.Sequential(model.layer1[4:6])
        model_list[4] = nn.Sequential(model.layer1[6:8])
        model_list[5] = nn.Sequential(model.layer1[8:9], model.layer2[:1])
        model_list[6] = nn.Sequential(model.layer2[1:3])
        model_list[7] = nn.Sequential(model.layer2[3:5])
        model_list[8] = nn.Sequential(model.layer2[5:7])
        model_list[9] = nn.Sequential(model.layer2[7:9])
        model_list[10] = nn.Sequential(model.layer3[:2])
        model_list[11] = nn.Sequential(model.layer3[2:4])
        model_list[12] = nn.Sequential(model.layer3[4:6])
        model_list[13] = nn.Sequential(model.layer3[6:8])
        model_list[14] = nn.Sequential(model.layer3[8:])
        model_list[15] = nn.Sequential(model.avgpool, Flatten(), model.fc)
    else:
        print('No modules selected...')
elif args.model == 'ResNet98':
    import ResNet_cifar as ResNet
    model = ResNet.resnet98(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:9])
        model_list[1] = nn.Sequential(model.layer2[9:], model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:12])
        model_list[1] = nn.Sequential(model.layer1[12:], model.layer2[:9])
        model_list[2] = nn.Sequential(model.layer2[9:], model.layer3[:5])
        model_list[3] = nn.Sequential(model.layer3[5:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:6])
        model_list[1] = nn.Sequential(model.layer1[6:13])
        model_list[2] = nn.Sequential(model.layer1[13:18], model.layer2[:2])
        model_list[3] = nn.Sequential(model.layer2[2:9])
        model_list[4] = nn.Sequential(model.layer2[9:16])
        model_list[5] = nn.Sequential(model.layer2[16:18], model.layer3[:5])
        model_list[6] = nn.Sequential(model.layer3[5:12])
        model_list[7] = nn.Sequential(model.layer3[12:], model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet110':
    import ResNet_cifar as ResNet
    model = ResNet.resnet110(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:9])
        model_list[1] = nn.Sequential(model.layer2[9:], model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:13])
        model_list[1] = nn.Sequential(model.layer1[13:], model.layer2[:9])
        model_list[2] = nn.Sequential(model.layer2[9:], model.layer3[:5])
        model_list[3] = nn.Sequential(model.layer3[5:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:10])
        model_list[1] = nn.Sequential(model.layer1[10:], model.layer2[:3])
        model_list[2] = nn.Sequential(model.layer2[3:14])
        model_list[3] = nn.Sequential(model.layer2[14:], model.layer3[:7])
        model_list[4] = nn.Sequential(model.layer3[7:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:6])
        model_list[1] = nn.Sequential(model.layer1[6:13])
        model_list[2] = nn.Sequential(model.layer1[13:18], model.layer2[:2])
        model_list[3] = nn.Sequential(model.layer2[2:9])
        model_list[4] = nn.Sequential(model.layer2[9:16])
        model_list[5] = nn.Sequential(model.layer2[16:18], model.layer3[:5])
        model_list[6] = nn.Sequential(model.layer3[5:12])
        model_list[7] = nn.Sequential(model.layer3[12:], model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet164':
    import ResNet_cifar as ResNet
    model = ResNet.resnet164(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:23])
        model_list[1] = nn.Sequential(model.layer1[23:], model.layer2, model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:15])
        model_list[1] = nn.Sequential(model.layer1[15:], model.layer2[:7])
        model_list[2] = nn.Sequential(model.layer2[7:], model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:12])
        model_list[1] = nn.Sequential(model.layer1[12:22])
        model_list[2] = nn.Sequential(model.layer1[22:], model.layer2[:16])
        model_list[3] = nn.Sequential(model.layer2[16:], model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:10])
        model_list[1] = nn.Sequential(model.layer1[10:], model.layer2[:3])
        model_list[2] = nn.Sequential(model.layer2[3:14])
        model_list[3] = nn.Sequential(model.layer2[14:], model.layer3[:7])
        model_list[4] = nn.Sequential(model.layer3[7:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 10:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:6])
        model_list[1] = nn.Sequential(model.layer1[6:12])
        model_list[2] = nn.Sequential(model.layer1[12:18])
        model_list[3] = nn.Sequential(model.layer1[18:])
        model_list[4] = nn.Sequential(model.layer2[:6])
        model_list[5] = nn.Sequential(model.layer2[6:12])
        model_list[6] = nn.Sequential(model.layer2[12:18])
        model_list[7] = nn.Sequential(model.layer2[18:])
        model_list[8] = nn.Sequential(model.layer3[:12])
        model_list[9] = nn.Sequential(model.layer3[12:], model.avgpool, Flatten(), model.fc)
elif args.model == 'ResNet1202':
    import ResNet_cifar as ResNet
    model = ResNet.resnet1202(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:50])
        model_list[1] = nn.Sequential(model.layer2[50:], model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:75])
        model_list[1] = nn.Sequential(model.layer1[75:], model.layer2[:50])
        model_list[2] = nn.Sequential(model.layer2[50:], model.layer3[:25])
        model_list[3] = nn.Sequential(model.layer3[25:], model.avgpool, Flatten(), model.fc)
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:60])
        model_list[1] = nn.Sequential(model.layer1[60:], model.layer2[:20])
        model_list[2] = nn.Sequential(model.layer2[20:80])
        model_list[3] = nn.Sequential(model.layer2[80:], model.layer3[:40])
        model_list[4] = nn.Sequential(model.layer3[40:], model.avgpool, Flatten(), model.fc)
elif args.model == 'WRN28_10':
    import WRN as WRN
    model = WRN.wrn28_10(num_classes=num_classes)
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.layer1, model.layer2[:1])
        model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.bn1, model.relu, model.avgpool, Flatten(), model.fc)
    if args.num_split == 3:
        model_list[0] = nn.Sequential(model.conv1, model.layer1)
        model_list[1] = nn.Sequential(model.layer2)
        model_list[2] = nn.Sequential(model.layer3, model.bn1, model.relu, model.avgpool, Flatten(), model.fc)
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.layer1[:3])
        model_list[1] = nn.Sequential(model.layer1[3:], model.layer2[:1])
        model_list[2] = nn.Sequential(model.layer2[1:])
        model_list[3] = nn.Sequential(model.layer3, model.bn1, model.relu, model.avgpool, Flatten(), model.fc)
    if args.num_split == 7:
        model_list[0] = nn.Sequential(model.conv1, model.layer1[:2])
        model_list[1] = nn.Sequential(model.layer1[2:])
        model_list[2] = nn.Sequential(model.layer2[:2])
        model_list[3] = nn.Sequential(model.layer2[2:])
        model_list[4] = nn.Sequential(model.layer3[:2])
        model_list[5] = nn.Sequential(model.layer3[2:])
        model_list[6] = nn.Sequential(model.bn1, model.relu, model.avgpool, Flatten(), model.fc)
elif args.model == 'resnext':
    if args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.resnext50_32x4d()
        if args.num_split == 2:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
            model_list[1] = nn.Sequential(model.layer2, model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        elif args.num_split == 3:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:2])
            model_list[1] = nn.Sequential(model.layer1[2:], model.layer2)
            model_list[2] = nn.Sequential(model.layer3, model.layer4, model.avgpool, Flatten(), model.fc)
        elif args.num_split == 4:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1[:1])
            model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[:2])
            model_list[2] = nn.Sequential(model.layer2[2:], model.layer3[:2])
            model_list[3] = nn.Sequential(model.layer3[2:], model.layer4, model.avgpool, Flatten(), model.fc)
    else:
        import ResNeXt as ResNeXt
        model = ResNeXt.ResNeXt29_32x4d(num_classes=num_classes)
        # model = ResNeXt.ResNeXt29_8x64d(num_classes=num_classes)
        if args.num_split == 2:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2[:1])
            model_list[1] = nn.Sequential(model.layer2[1:], model.layer3, model.avgpool, Flatten(), model.fc)
        elif args.num_split == 3:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1)
            model_list[1] = nn.Sequential(model.layer2)
            model_list[2] = nn.Sequential(model.layer3, model.avgpool, Flatten(), model.fc)
        elif args.num_split == 4:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1[:1])
            model_list[1] = nn.Sequential(model.layer1[1:], model.layer2[:2])
            model_list[2] = nn.Sequential(model.layer2[2:], model.layer3[:2])
            model_list[3] = nn.Sequential(model.layer3[2:], model.avgpool, Flatten(), model.fc)
        elif args.num_split == 11:
            model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu)
            model_list[1] = nn.Sequential(model.layer1[0])
            model_list[2] = nn.Sequential(model.layer1[1])
            model_list[3] = nn.Sequential(model.layer1[2])
            model_list[4] = nn.Sequential(model.layer2[0])
            model_list[5] = nn.Sequential(model.layer2[1])
            model_list[6] = nn.Sequential(model.layer2[2])
            model_list[7] = nn.Sequential(model.layer3[0])
            model_list[8] = nn.Sequential(model.layer3[1])
            model_list[9] = nn.Sequential(model.layer3[2], model.avgpool)
            model_list[10] = nn.Sequential(Flatten(), model.fc)
elif args.model == 'MobileNet':
    import mobilenet as MobileNet
    model = MobileNet.MobileNetv2(num_classes=num_classes)

    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.conv1, model.bn1, model.relu, model.layers[:6])
        model_list[1] = nn.Sequential(model.layers[6:], model.avgpool, Flatten(), model.fc)
elif args.model == 'vgg11_bn':
    import torchvision.models as models
    model = models.vgg11_bn()
    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.features[:8])
        model_list[1] = nn.Sequential(model.features[8:15])
        model_list[2] = nn.Sequential(model.features[15:22])
        model_list[3] = nn.Sequential(model.features[22:],model.avgpool, Flatten(), model.classifier)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.features[:4])
        model_list[1] = nn.Sequential(model.features[4:8])
        model_list[2] = nn.Sequential(model.features[8:11])
        model_list[3] = nn.Sequential(model.features[11:15])
        model_list[4] = nn.Sequential(model.features[15:18])
        model_list[5] = nn.Sequential(model.features[18:22])
        model_list[6] = nn.Sequential(model.features[22:])
        model_list[7] = nn.Sequential(model.avgpool, Flatten(), model.classifier)
elif args.model == 'vgg11':
    if args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.vgg11()
        if args.resume:
            print('loading models!')
            model.load_state_dict(checkpoint['state_dict'])
            if args.num_split == 4:
                model_list[0] = nn.Sequential(model.features[:8])
                model_list[1] = nn.Sequential(model.features[8:15])
                model_list[2] = nn.Sequential(model.features[15:22])
                model_list[3] = nn.Sequential(model.features[22:], Flatten(), model.classifier)
            if args.num_split == 6:
                model_list[0] = nn.Sequential(model.features[:4])
                model_list[1] = nn.Sequential(model.features[4:10])
                model_list[2] = nn.Sequential(model.features[10:17])
                model_list[3] = nn.Sequential(model.features[17:24])
                model_list[4] = nn.Sequential(model.features[24:])
                model_list[5] = nn.Sequential(Flatten(), model.classifier)
        else:
            from vggnet import VGG as models
            model = models('VGG11')
            if args.num_split == 4:
                model_list[0] = nn.Sequential(model.features[:6])
                model_list[1] = nn.Sequential(model.features[6:15])
                model_list[2] = nn.Sequential(model.features[15:])
                model_list[3] = nn.Sequential(Flatten(), model.classifier)
            if args.num_split == 9:
                model_list[0] = nn.Sequential(model.features[:3])
                model_list[1] = nn.Sequential(model.features[3:6])
                model_list[2] = nn.Sequential(model.features[6:8])
                model_list[3] = nn.Sequential(model.features[8:11])
                model_list[4] = nn.Sequential(model.features[11:13])
                model_list[5] = nn.Sequential(model.features[13:16])
                model_list[6] = nn.Sequential(model.features[16:18])
                model_list[7] = nn.Sequential(model.features[18:])
                model_list[8] = nn.Sequential(Flatten(), model.classifier)
elif args.model == 'VGG19':
    import torchvision.models as models
    model = models.vgg19()
    if args.num_split == 2:
        model_list[0] = nn.Sequential(model.features[:12])
        model_list[1] = nn.Sequential(model.features[12:], model.avgpool, Flatten(), model.classifier)
elif args.model == 'squeezenet':
    import torchvision.models as models
    model = models.squeezenet1_0()
    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 5:
        model_list[0] = nn.Sequential(model.features[:4])
        model_list[1] = nn.Sequential(model.features[4:7])
        model_list[2] = nn.Sequential(model.features[7:9])
        model_list[3] = nn.Sequential(model.features[9:12])
        model_list[4] = nn.Sequential(model.features[12:], model.classifier, Flatten())
elif args.model == 'densenet121':
    import torchvision.models as models
    model = models.densenet121()
    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.features[:3])
        model_list[1] = nn.Sequential(model.features[3:6])
        model_list[2] = nn.Sequential(model.features[6:9])
        model_list[3] = nn.Sequential(model.features[9:12], nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)), Flatten(), model.classifier)
elif args.model == 'shufflenet':
    import torchvision.models as models
    model = models.shufflenet_v2_x0_5()
    class Mean2_3(nn.Module):
        def forward(self, x):
            return x.mean([2, 3])

    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.conv1, model.maxpool, model.stage2)
        model_list[1] = nn.Sequential(model.stage3)
        model_list[2] = nn.Sequential(model.stage4)
        model_list[3] = nn.Sequential(model.conv5, Mean2_3(), model.fc)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.maxpool, model.stage2)
        model_list[1] = nn.Sequential(model.stage3)
        model_list[2] = nn.Sequential(model.stage4)
        model_list[3] = nn.Sequential(model.conv5, Mean2_3(), model.fc)
elif args.model == 'mnasnet':
    import torchvision.models as models
    model = models.mnasnet0_5()
    class Mean2_3(nn.Module):
        def forward(self, x):
            return x.mean([2, 3])

    if args.resume:
        print('loading models!')
        model.load_state_dict(checkpoint['state_dict'])
    if args.num_split == 4:
        model_list[0] = nn.Sequential(model.layers[:4])
        model_list[1] = nn.Sequential(model.layers[4:8])
        model_list[2] = nn.Sequential(model.layers[8:12])
        model_list[3] = nn.Sequential(model.layers[12:], Mean2_3(), model.classifier)
    if args.num_split == 8:
        model_list[0] = nn.Sequential(model.conv1, model.maxpool, model.stage2)
        model_list[1] = nn.Sequential(model.stage3)
        model_list[2] = nn.Sequential(model.stage4)
        model_list[3] = nn.Sequential(model.conv5, Mean2_3(), model.fc)
else:
    print('No specified models!!')

if not args.backprop:
    if args.dataset == 'imagenet':
        images = torch.randn(1, 3, 224, 224).to(device[0])
    elif args.dataset.startswith('CIFAR'):
        images = torch.randn(1, 3, 32, 32).to(device[0])
    else:
        print('other datasets!')
    with torch.no_grad():
        model = model.to(device[0])
        model.eval()
        for m in model_list:
            model_list[m].eval()

        outputs1 = model(images)
        outputs2 = images

        for m in model_list:
            outputs2 = model_list[m](outputs2)

        # print(outputs1-outputs2)
        dif = outputs1 - outputs2
    if dif.sum() == 0:
        print('split valid!')
    else:
        print('split invalid!')
        sys.exit()

if args.backprop:
    from torchgpipe import GPipe
    model_ = []
    gpus = []
    m_ = []
    for item in range(args.num_split):
        model_.append(model_list[item])
        m_.append(1)
        gpus.append(0)
    model = nn.Sequential(*model_)
    if args.mulgpu:
        gpus = None
    model = GPipe(model, balance=m_, chunks=4, devices=gpus)

optimizer = {}
scheduler = {}
for m in model_list:
    model_list[m] = model_list[m].to(device[m])
    if args.optim == 'adam':
        optimizer[m] = optim.Adam(model_list[m].parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        if args.resume:
            optimizer[m] = optimizer_save[m]
        else:
            optimizer[m] = optim.SGD(model_list[m].parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        print('no optimizer found!')
    scheduler[m] = LRS.MultiStepLR(optimizer[m], milestones=args.lr_decay_milestones, gamma=args.lr_decay_fact)
if args.backprop:
    print('backprop training!')
    local_classifier = {}
    for m in range(args.num_split):
        local_classifier[m] = None
else:
    print('Breaking {} into {} pieces.'.format(args.model, args.num_split))

module = {}
for m in range(args.num_split):
    module[m] = BuildDG(model_list[m], optimizer[m], scheduler[m], m)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.type == 'DGL':
    # build local classifiers
    local_classifier = {}
    images = torch.randn(2,input_ch,input_dim,input_dim).to(device[0])
    outputs = images
    for m in range(args.num_split-1):
        outputs = outputs.to(device[m])
        outputs = model_list[m](outputs)
        tmp = auxillary_dgl(input_features=outputs.size(1), in_size=outputs.size(2),
                                        num_classes=num_classes, n_lin=3, mlp_layers=3, batchn=True).to(device[m])
        optimizer = optim.SGD(tmp.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        aux_outputs = tmp(outputs)
        scheduler = LRS.MultiStepLR(optimizer, milestones=args.lr_decay_milestones, gamma=args.lr_decay_fact)
        local_classifier[m] = BuildAuxNet(tmp, optimizer, scheduler, m)
        local_classifier[m].scheduler.step()
    local_classifier[args.num_split-1] = None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




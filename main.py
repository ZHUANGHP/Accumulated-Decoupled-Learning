import os
import shutil
import time
import math

from parsers import get_string
from datasets import train_loader, test_loader
from models import model, module, scheduler, optimizer, device, best_prec1, args
import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import threading as thread
import time
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
import queue
from collections import deque
from datetime import datetime
if args.type == 'DGL':
    from models import local_classifier

string_print = get_string(args)
now = datetime.now()
string = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(
    now.minute) + '-' + str(now.second) + '-' + str(np.random.randint(1000))

if args.save:
    print(args.resume)
    if args.resume:
        print('loading model saving directory!')
        if 'model_best.pth.tar' in args.resume:
            dirname = args.resume.replace('model_best.pth.tar', '')
        elif 'checkpoint.pth.tar' in args.resume:
            dirname = args.resume.replace('checkpoint.pth.tar', '')
        print(dirname)
    else:
        print('creating model saving dir!')
        dirname = ''
        dirname = os.path.join(dirname, 'save_model')
        if args.backprop:
            dirname = os.path.join(dirname, 'BP')
        else:
            dirname = os.path.join(dirname, args.type)
        dirname = os.path.join(dirname, args.dataset)
        dirname = os.path.join(dirname, args.model)
        dirname = dirname.replace("\\", "/")
        dirname = os.path.join(dirname, string)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(os.path.join(dirname, 'args.txt'), 'w') as file:
            file.write(str(args))
# define writer
if args.writer:
    from torch.utils.tensorboard import SummaryWriter
    if args.backprop:
        string2 = 'BP_' + args.dataset + '_' + args.model + '_'
    else:
        string2 = args.type + args.mode + '_' + args.dataset + '_' + args.model + '_' + 's' + str(args.num_split) + \
                  'b' + str(args.batch_size) + 'a' + str(args.ac_step) + '_'
    print(string2)
    writer = SummaryWriter(args.type + '/' + string2 + string)
    args.dirname = args.type + '/' + string2 + string
    with open(os.path.join(args.dirname, 'args.txt'), 'w') as file:
        file.write(str(args))
else:
    writer = None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(dirname, filename)
    print(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


def trainfdg(model, images, labels):
    # name = thread.currentThread().getName()
    if not model.last_layer:
        if images is not None:
            model.train()
            if args.mode == 'A':
                outputs = model(images)
                receive_grad[model.module_num] = model.backward()
                if model.update_count >= args.ac_step:
                    model.step()
                    model.zero_grad()
                    model.update_count = 0
                model.output.append(outputs)
            elif args.mode == 'B':
                receive_grad[model.module_num] = model.backward()
                if model.update_count >= args.ac_step:
                    model.step()
                    model.zero_grad()
                    model.update_count = 0
                outputs = model(images)
                model.output.append(outputs)
            else:
                print('incorrect mode!')
            #model.output.append(outputs.detach())
            if not model.first_layer:
                model.input.append(images)
                input_images = model.input.popleft()
                if input_images is None:
                    print('no input gradients obtained in module {}'.format(model.module_num))
                else:
                    model.input_grad = input_images.grad
    elif model.last_layer:
        if images is not None and labels is not None:
            model.train()            
            if args.mode == 'A':
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                model.update_count += 1
                if receive_grad[model.module_num-1] is True:
                    if model.update_count >= args.ac_step:
                        model.step()
                        model.zero_grad()
                        model.update_count = 0
                else:
                    pass
            elif args.mode == 'B':
                if receive_grad[model.module_num-1] is True:
                    if model.update_count >= args.ac_step:
                        model.step()
                        model.zero_grad()
                        model.update_count = 0
                else:
                    pass
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                model.update_count += 1
            else:
                print('incorrect mode!')
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            model.input_grad = images.grad
            model.acc = prec1
            model.acc5 = prec5
            model.loss = loss


def trainfr(model, images, labels, module_input):
    if not model.last_layer and images is not None:
        model.train()
        input_images = model.input[0]
        if input_images is not None:
            input_images.retain_grad()
        if input_images is not None:
            outputs = model(input_images)
            model.zero_grad()
            if model.dg is not None:
                outputs.backward(model.dg)
            else:
                print('No delayed gradients in module {}'.format(model.module_num))
            model.step()
        else:
            print('no backwarding.')
        '''
        if model.input_grad is None:
            print('no input gradients obtained in module {}'.format(model.module_num))
        '''
        model.input_grad = input_images.grad if input_images is not None else None

        if model.input_grad is None:
            print('no input gradients obtained in module {}'.format(model.module_num))

    elif model.last_layer:
        if labels is not None:
            model.train()
            outputs = model.get_output()
            loss = F.cross_entropy(outputs, labels)
            module_input.retain_grad()
            model.zero_grad()
            loss.backward()
            model.step()
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            model.input_grad = module_input.grad
            model.acc = prec1
            model.acc5 = prec5
            model.loss = loss


def trainddg(model, images, labels):
    if not model.last_layer:
        model.train()
        if not model.first_layer:
            input_images = model.input.popleft()
            if input_images is not None:
                input_images.retain_grad()
        model.zero_grad()
        model.backward()
        model.step()
        if not model.first_layer:
            if input_images is not None:
                model.input_grad = input_images.grad if input_images is not None else None
                if input_images.grad is None:
                    print('no input gradients obtained in module {}'.format(model.module_num))

    elif model.last_layer:
        if labels is not None:
            model.train()
            outputs = model.get_output()
            module_input = model.input.popleft()
            module_input.retain_grad()
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            model.step()

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            model.input_grad = module_input.grad
            model.acc = prec1
            model.acc5 = prec5
            model.loss = loss


def traindgl(model, local_cls, images, labels):
    if not model.last_layer:
        if images is not None and labels is not None:
            # primal networks
            model.train()
            local_cls.model.train()
            outputs = model(images)
            # aux networks
            aux_input = Variable(outputs.data, requires_grad=True)
            aux_outputs = local_cls(aux_input)
            labels_aux = labels
            loss_aux = F.cross_entropy(aux_outputs, labels_aux)
            # local performance
            _, predicted = torch.max(aux_outputs.data, 1)
            correct = (predicted == labels_aux).sum().item()
            acc = correct / images.size(0)
            #model.acc = acc
            #top1.update(acc, labels.size(0))
            #model.loss = loss_aux.item()

            # reset gradients
            model.zero_grad()
            local_cls.zero_grad()
            # local bp
            loss_aux.backward()
            # outputs.backward(aux_input.grad)
            outputs.backward(aux_input.grad)
            # update
            local_cls.step()
            model.step()
            # store activations
            model.output.append(outputs.detach())
    else:
        if images is not None and labels is not None:
            model.train()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            # training results
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            model.acc = prec1
            model.acc5 = prec5
            model.loss = loss
            # bp and update
            model.zero_grad()
            loss.backward()
            model.step()


def train_bp(model, images, labels):
    model.train()
    outputs = model(images.to(device[0]))
    loss = F.cross_entropy(outputs, labels)

    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))

    for m in range(args.num_split):
        module[m].zero_grad()
    loss.backward()
    for m in range(args.num_split):
        module[m].optimizer.step()
    
    return prec1, prec5, loss


def train(train_loader, module, epoch):

    for m in range(args.num_split):
        receive_grad[m] = False
    if args.pbar:
        pbar = tqdm(enumerate(train_loader), desc='Training Epoch {}/{}'.format(str(epoch + 1), args.epochs),
                    total=len(train_loader), unit='batch')
    else:
        pbar = enumerate(train_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    com_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for i, (images, labels) in pbar:
        lr = adjust_learning_rate(module=module, epoch=epoch, step=i, len_epoch=len(train_loader))
        if args.prof:
            if i > 10:
                break
        data_time.update(time.time() - end)
        images = images.to(device[0], non_blocking=True)
        labels = labels.to(device[0], non_blocking=True)

        # newest batch input
        input_info[0] = images
        labels_q.put(labels)

        if args.backprop:
            labels = labels.to(device[0])
            prec1, prec5, loss = train_bp(model, images.to(device[0]), labels.to(device[args.num_split - 1]))
            losses.update(to_python_float(loss), images.size(0))
            top1.update(to_python_float(prec1), images.size(0))
            top5.update(to_python_float(prec5), images.size(0))
        elif args.type == 'FDG' or args.type == 'ADL':
            processes = []
            if args.mulgpu:
                # porblems could happen if run in threads
                for m in range(args.num_split):
                    p = thread.Thread(target=trainfdg, args=(module[m], input_info[m], target_info[m]))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for m in range(args.num_split):
                    trainfdg(module[m], input_info[m], target_info[m])

            # passing inputs
            time_communication = time.time()
            for m in reversed(range(1, args.num_split)):
                tmp = module[m - 1].get_output()
                input_info[m] = Variable(tmp.detach().clone().to(device[m]), requires_grad=True) if tmp is not None else None

            for m in range(args.num_split - 1):
                module[m].dg = module[m + 1].input_grad.clone().to(device[m]) if module[m + 1].input_grad is not None else None
            com_time.update(time.time() - time_communication)
            labels_last = labels_q.get()
            target_info[last_idx] = labels_last.to(device[last_idx]) if labels_last is not None else None
            if module[last_idx].acc != 0:
                top1.update(to_python_float(module[last_idx].acc), input_info[last_idx].size(0))
                top5.update(to_python_float(module[last_idx].acc5), input_info[last_idx].size(0))
            if module[last_idx].loss != 0:
                losses.update(to_python_float(module[args.num_split - 1].loss), input_info[last_idx].size(0))
        elif args.type == 'FR':
            processes = []
            outputs = images
            for m in range(args.num_split):
                module_input = Variable(outputs.data, requires_grad=True)
                module_input = module_input.to(device[m])
                module[m].input.append(module_input)
                outputs = module[m](module_input)
                module[m].output.append(outputs.data)
            module[args.num_split - 1].output.append(outputs)
            ######################################
            target_info[args.num_split - 1] = labels.to(device[args.num_split - 1])
            ######################################
            # porblems could happen if run in threads
            if args.mulgpu:
                for m in range(args.num_split):
                    p = thread.Thread(target=trainfr, args=(module[m], input_info[m], target_info[m], module_input))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for i in range(args.num_split):
                    trainfr(module[i], input_info[i], target_info[i], module_input)
            time_communication = time.time()
            for i in reversed(range(1, args.num_split)):
                tmp = module[i - 1].get_output()
                input_info[i] = Variable(tmp.detach().clone().to(device[i]),
                                         requires_grad=True) if tmp is not None else None

            for i in range(args.num_split - 1):
                module[i].dg = module[i + 1].input_grad.clone().to(device[i]) if module[
                                                                                     i + 1].input_grad is not None else None

            target_info[last_idx] = labels_q.get()
            com_time.update(time.time() - time_communication)
            if module[last_idx].acc != 0:
                top1.update(to_python_float(module[last_idx].acc), args.batch_size)
                top5.update(to_python_float(module[last_idx].acc5), args.batch_size)
            if module[last_idx].loss != 0:
                losses.update(to_python_float(module[args.num_split - 1].loss), args.batch_size)
        elif args.type == 'DDG':
            outputs = images
            for m in range(args.num_split):
                module_input = Variable(outputs.data, requires_grad=True)
                module_input = module_input.to(device[m])
                # outputs = outputs.to(device[m])
                module[m].input.append(module_input)
                outputs = module[m](module_input)
                module[m].output.append(outputs)

            target_info[args.num_split - 1] = labels.to(device[args.num_split - 1])
            ######################################
            if args.mulgpu:
                processes = []
                # porblems could happen if run in threads
                for m in range(args.num_split):
                    p = thread.Thread(target=trainddg, args=(module[m], input_info[m], target_info[m]))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for i in range(args.num_split):
                    trainddg(module[i], input_info[i], target_info[i])

            # passing inputs
            time_communication = time.time()
            for i in reversed(range(1, args.num_split)):
                tmp = module[i - 1].get_output()
                input_info[i] = Variable(tmp.detach().to(device[i]), requires_grad=True) if tmp is not None else None

            for i in range(args.num_split - 1):
                module[i].dg = module[i + 1].input_grad.to(device[i]) if module[i + 1].input_grad is not None else None
            com_time.update(time.time() - time_communication)

            if module[last_idx].acc != 0:
                top1.update(to_python_float(module[last_idx].acc), input_info[last_idx].size(0))
                top5.update(to_python_float(module[last_idx].acc5), input_info[last_idx].size(0))
            if module[last_idx].loss != 0:
                losses.update(to_python_float(module[args.num_split - 1].loss), input_info[last_idx].size(0))
        elif args.type == 'DGL':
            input_info[0] = images.to(device[0])
            labels_dq.append(labels)
            for i in range(args.num_split):
                target_info[i] = labels_dq[args.num_split - i - 1].to(device[i]) if labels_dq[
                                                                                       args.num_split - i - 1] is not None else None
                target_info[i] = target_info[i].to(device[i]) if target_info[i] is not None else None

            if not args.mulgpu:
                # train the modules separately
                for i in range(args.num_split):
                    traindgl(module[i], local_classifier[i], input_info[i], target_info[i])
                    # print(module[0].acc)
            else:
                processes = []
                # porblems could happen if run in threads
                for m in range(args.num_split):
                    p = thread.Thread(target=traindgl,
                                      args=(module[m], local_classifier[m], input_info[m], target_info[m]))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

            # passing inputs
            for i in reversed(range(1, args.num_split)):
                tmp = module[i-1].get_output()
                #tmp = module[i - 1].output
                input_info[i] = Variable(tmp.clone().to(device[i]), requires_grad=True) if tmp is not None else None

            # record training results
            if module[last_idx].acc != 0:
                top1.update(to_python_float(module[last_idx].acc), input_info[last_idx].size(0))
                top5.update(to_python_float(module[last_idx].acc5), input_info[last_idx].size(0))
            if module[last_idx].loss != 0:
                losses.update(to_python_float(module[args.num_split - 1].loss), input_info[last_idx].size(0))
        else:
                print('No learning method is specified!')
                break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.pbar:
            pbar.set_postfix_str('ACC@1:' + str(round(top1.avg, 3)) +
                                 ' ACC@5:' + str(round(top5.avg, 3)) +
                                 ' Loss:' + str(round(losses.avg, 3)) +
                                 ' com T:' + str(round(com_time.val, 3)) +
                                 ' data T:' + str(round(data_time.val, 3)) +
                                 ' batch T:' + str(round(batch_time.val, 3)) +
                                 ' lr:' + str(round(lr, 3))
                                 )
    if args.writer:
        writer.add_scalar(string_print + '/train_top1', top1.avg, epoch + 1)
        writer.add_scalar(string_print + '/train_top5', top5.avg, epoch + 1)
        writer.add_scalar(string_print + '/train_loss', losses.avg, epoch + 1)
    return [top1.avg, top5.avg]


def validate(test_loader, module, epoch):
    print(string_print)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    for m in range(args.num_split):
        module[m].model.eval()
    if args.pbar:
        pbar = tqdm(enumerate(test_loader), desc='Testing Epoch {}/{}'.format(str(epoch + 1), args.epochs), total=len(test_loader), unit='batch')
    else:
        pbar = enumerate(test_loader)
    with torch.no_grad():
        end = time.time()

        for i, (images, labels) in pbar:
            if args.prof:
                if i > 10:
                    break
            '''if mulgpu watch out for the model distributing it into several GPUs!'''
            outputs = images
            labels = labels.to(device[args.num_split - 1])
            for m in range(args.num_split):
                outputs = outputs.to(device[m])
                outputs = module[m](outputs)
            loss = F.cross_entropy(outputs, labels)

            losses.update(to_python_float(loss), images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1,5))
            top1.update(to_python_float(prec1), images.size(0))
            top5.update(to_python_float(prec5), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.pbar:
                pbar.set_postfix_str('ACC@1:' + str(round(top1.avg, 3)) +
                                     ' ACC@5:' + str(round(top5.avg, 3)) +
                                     ' Loss:' + str(round(losses.avg, 3)) +
                                     ' batch T:' + str(round(batch_time.val, 3))
                                     )
    
    if args.writer:
        writer.add_scalar(string_print + '/val_top1', top1.avg, epoch + 1)
        writer.add_scalar(string_print + '/val_top5', top5.avg, epoch + 1)
        writer.add_scalar(string_print + '/val_loss', losses.avg, epoch + 1)
    return top1.avg


def main():
    global best_prec1, receive_grad, args
    for m in range(args.num_split):
        receive_grad[m] = False
    if args.evaluate:
        prec1 = validate(test_loader, module, 0)
        return

    print('Training begins')

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, module, epoch)
        prec1 = validate(test_loader, module, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.save:
            optimizer_save = []
            for m in range(args.num_split):
                optimizer_save.append(module[m].optimizer)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer_save,
            }, is_best)

    if args.writer:
        with open(os.path.join(args.dirname, 'last epoch-' + str(round(prec1,2)) + '.txt'), 'w') as file:
            file.write(str(prec1))
        with open(os.path.join(args.dirname, 'bst-' + str(round(best_prec1,2)) + '.txt'), 'w') as file:
            file.write(str(best_prec1))
        os.rename(args.type + '/' + string2 + string, args.type + '/' + string2 + string + '_' + str(round(prec1,2)) + '-' + str(round(best_prec1,2)))


def adjust_learning_rate(module, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    for i in range(len(args.lr_decay_milestones)):
        if epoch >= args.lr_decay_milestones[-1]:
            scaling = len(args.lr_decay_milestones)
            break
        elif epoch < args.lr_decay_milestones[i]:
            scaling = i
            break
    lr = args.lr * 10**(-scaling)
    """Warmup"""
    if epoch < args.warm_up_epochs:
        lr = 0.01*args.lr + (args.lr - 0.01*args.lr)*(step + 1 + epoch*len_epoch)/(args.warm_up_epochs * len_epoch)
    for m in range(args.num_split):
        for param_group in module[m].optimizer.param_groups:
            param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    best_prec1 = best_prec1
    input_info = {}
    target_info = {}
    last_idx = args.num_split - 1
    labels_q = queue.Queue()
    labels_dq = deque(maxlen=args.num_split)
    receive_grad = []
    for i in range(args.num_split):
        input_info[i], target_info[i] = None, None
        labels_dq.append(None)
        receive_grad.append(False)
    for i in range(args.num_split-2):
        labels_q.put(None)
    main()
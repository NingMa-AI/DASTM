# coding=utf-8
import random

from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from nturgbd_dataset import NTU_RGBD_Dataset
from protonet import ProtoNet
from parser_util import get_parser
from utils import load_data, get_para_num, setup_seed,getAvaliableDevice

from tqdm import tqdm
import numpy as np
import torch
import pickle
import os
import time
import gl
import warnings
from utils import *

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, data_list, mode):
    debug = False
    dataset = NTU_RGBD_Dataset(mode=mode, data_list=data_list, debug=debug, extract_frame=opt.extract_frame)
    n_classes = len(np.unique(dataset.label))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
        iters = opt.train_iterations
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
        iters = opt.test_iterations

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iters)

def init_dataloader(opt, data_list, mode):
    dataset = init_dataset(opt, data_list, mode)
    sampler = init_sampler(opt, dataset.label, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    return dataloader

def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    model = ProtoNet(opt).to(gl.device)
    if opt.model == 1:
        model_path = os.path.join(opt.experiment_root, 'best_model.pth')
        # print('model_path', model_path)
        model.load_state_dict(torch.load(model_path))
    # print(get_para_num(model))
    return model

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate, weight_decay=5e-4)

    return optimizer

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    if opt.lr_flag == 'reduceLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-5)
    elif opt.lr_flag == 'stepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=opt.lr_scheduler_gamma,
                                                       step_size=opt.lr_scheduler_step)

    return lr_scheduler

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, test_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    import json
    with open(os.path.join(opt.experiment_root, 'opt.json'), 'w') as f:
        j = vars(opt)
        json.dump(j, f)
        f.write('\n')

    if val_dataloader is None:
        best_state = None

    best_acc = 0
    last_acc = 0
    acc_reduce_num = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    trace_file = os.path.join(opt.experiment_root, 'trace.txt')

    start_epoch = 0

    patience=0

    for epoch in range(start_epoch, opt.epochs):
        gl.epoch = epoch
        gl.iter = 0
        # print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        lr = opt.learning_rate
        train_acc = []
        reg_loss = []
        train_loss = []
        for batch in tqdm(tr_iter):
        # for batch in tr_iter:
            optim.zero_grad()
            gl.mod = 'train'
            x, y = batch
            x, y = x.to(gl.device).float(), y.to(gl.device)
            model_output = model(x)
            loss, acc, reg = model.loss(model_output, y, opt.num_support_tr,opt.dtw)

            train_loss.append(loss.item())
            train_acc.append(acc.item())
            reg_loss.append(reg.item())

            loss.backward()
            optim.step()

        avg_loss = np.mean(train_loss)
        avg_reg = np.mean(reg_loss)

        avg_acc = np.mean(train_acc)

        t_loss, t_acc = avg_loss, avg_acc
        string = 'train loss: {}, classfier loss:{} reg loss: {}, train Acc: {}'.format(avg_loss, avg_loss - avg_reg, avg_reg, avg_acc)

        if opt.lr_flag == 'reduceLR':
            lr_scheduler.step(avg_loss)
        elif opt.lr_flag == 'stepLR':
            lr_scheduler.step()

        lr = optim.state_dict()['param_groups'][0]['lr']

        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)

        model.eval()

        val_loss = []
        val_acc = []

        for batch in tqdm(val_iter):
        # for batch in val_iter:
            x, y = batch
            x, y = x.to(gl.device).float(), y.to(gl.device)
            gl.mod = 'val'
            model_output = model(x)
            loss, acc, reg = model.loss(model_output, target=y, n_support=opt.num_support_val,dtw=opt.dtw)

            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_loss = np.mean(val_loss)

        avg_acc = np.mean(val_acc)

        # if acc reduce 10 times, break
        if last_acc == 0:
            last_acc = avg_acc
        else:
            if last_acc >= avg_acc:
                acc_reduce_num += 1
            else:
                acc_reduce_num = 0
            last_acc = avg_acc
        if acc_reduce_num >= 10:
            print('acc already reduce more than 10 times!!  end training...')
            break

        v_loss, v_acc = avg_loss, avg_acc
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        string_val = 'val loss: {}, val acc: {}{} lr:{}'.format(avg_loss, avg_acc, postfix, lr)
        print(string + '\t' + string_val)
        with open(trace_file, 'a') as f:
            f.write(string + '\t' + string_val)
            f.write('\n')


        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            patience=0
            best_acc = avg_acc
            best_state = model.state_dict()
        else :
            patience+=1

        if patience >40:
            break
    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    print('testing model...')
    avg_acc = list()
    trace_file = os.path.join(opt.experiment_root, 'test.txt')

    n_class_val, n_query_val = opt.classes_per_it_val, opt.num_query_val

    for epoch in range(10):
        print('=== Epoch: {} ==='.format(epoch))
        model.eval()
        gl.epoch = epoch
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(gl.device).float(), y.to(gl.device)
            model_output = model(x)
            _, acc, _ = model.loss(model_output, target=y, n_support=opt.num_support_val,dtw=opt.dtw)
            avg_acc.append(acc.item())

        # print('test avg_acc', np.mean(avg_acc))

    avg_acc = np.mean(avg_acc)
    with open(trace_file, 'a') as f:
        f.write('test acc: {}'.format(avg_acc))
        f.write('\n')
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    options.experiment_root=os.path.join(options.experiment_root, "seed_"+str(str(options.manual_seed)),
    "_dataset"+str(options.dataset),"_back"+str(options.backbone),"_reg"+str(options.reg_rate)+"_att"+str(options.SA)+"_dtw"+str(options.dtw))

    options.cuda=True
    options.device=str(1)

    if options.debug == 1:
        gl.debug = True

    device = 'cuda:{}'.format(options.device) if torch.cuda.is_available() and options.cuda else 'cpu'
    gl.device = device

    gl.gamma = options.gamma
    options.experiment_root = "../log/"+options.experiment_root
    gl.experiment_root=options.experiment_root
    gl.reg_rate = options.reg_rate
    gl.threshold = options.thred
    gl.backbone = options.backbone
    gl.dataset = options.dataset
    gl.SA = options.SA

    if not os.path.exists(gl.experiment_root):
        os.makedirs(gl.experiment_root)


    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    setup_seed(options.manual_seed)

    data_list = []

    tr_dataloader = init_dataloader(options, data_list, 'train')
    val_dataloader = init_dataloader(options, data_list, 'val')
    test_dataloader = init_dataloader(options, data_list, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)

    if options.mode == 'train':
        res = train(opt=options,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
        best_state, best_acc = res

        model.load_state_dict(best_state)
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model)
    elif options.mode == 'test':
        print('Testing with best model..')
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model)

if __name__ == '__main__':
    main()

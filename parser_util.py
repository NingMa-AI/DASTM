# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset',
                        type=str,
                        help='type of dataset',
                        default='ntu-T', choices=["ntu-S","ntu-T","kinetics"])

    parser.add_argument('-mode', '--mode',
                        type=str,
                        help='mode',
                        default='train')

    parser.add_argument('-reg', '--reg_rate',
                        type=float,
                        help='reg',
                        default=0.1)

    parser.add_argument('-reg_thred', '--thred',
                        type=int,
                        help='threshold',
                        default=3)

    parser.add_argument('-gama', '--gamma',
                        type=float,
                        help='reg',
                        default=0.01)

    parser.add_argument('-dbg', '--debug',
                        type=int,
                        help='debug to save x and sim_tenor',
                        default=0)

    parser.add_argument('-model', '--model',
                        type=int,
                        help='use or not best model',
                        default=0)

    parser.add_argument('-backbone', '--backbone',
                        type=str,
                        help='backbone type st_gcn, 2s_AGCN, ms_g3d',
                        default='stgcn')

    parser.add_argument('-extrf', '--extract_frame',
                        type=int,
                        help='is or not extract frame',
                        default=1)

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='test')

    parser.add_argument('-d', '--device',
                        type=int,
                        help='GPU device',
                        default=0)
                        
    parser.add_argument('-dt', '--dtw',
                        type=int,
                        help='if using dtw',
                        default=1)

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrf', '--lr_flag',
                        type=str,
                        help='lr_scheduler type',
                        default='reduceLR')

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--train_iterations',
                        type=int,
                        help='number of episodes per epoch, default=1000',
                        default=1000)
    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=5',
                        default=5)
    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=1',
                        default=1)
    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=10',
                        default=10)
    parser.add_argument('-test_its', '--test_iterations',
                        type=int,
                        help='number of episodes per epoch, default=500',
                        default=500)
    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)
    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=1',
                        default=1)
    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=10',
                        default=10)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=2022)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    parser.add_argument('--SA',
                        type=int,
                        help='attention',
                        default=0)
    return parser

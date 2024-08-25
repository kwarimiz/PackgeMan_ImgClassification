# config_parser.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='命令行参数')
    parser.add_argument('--start_epoch', type=int, metavar='', help='the number of the start epoch', default=0)
    parser.add_argument('-e', '--end_epoch', type=int, metavar='', help='the number of epochs', default=100)
    parser.add_argument('-d', '--dataset', type=str, metavar='', help='the name of dataset', default='8_class_select')
    parser.add_argument('-b', '--batch_size', type=int, metavar='', help='the number of batch size', default=64)
    parser.add_argument('-n', '--net', type=str, metavar='', help='network', default='resnet34')
    parser.add_argument('-l', '--loss_function', type=str, metavar='', help='loss function', default='CrossEntropyLoss')
    parser.add_argument('-r', '--result_folder', type=str, metavar='', help='the path to save result')
    # parser.add_argument('-s', '--save_per_epoch', type=int, metavar='', help='save per epoch', default=5)
    parser.add_argument('-w', '--weight', type=str, metavar='', help='the weight name', default='pretrained')
    parser.add_argument('-o', '--optimizer', type=str, metavar='', help='the optimizer', default='Adam')
    parser.add_argument('-s', '--scheduler', type=str, metavar='', help='the scheduler', default='Cos')
    parser.add_argument('--sampler', type=str, metavar='', help='the dataloader sampler', default=None)
    parser.add_argument('--result_root', type=str, metavar='', help='the root path to save result', default='result_pretrain')

    return parser.parse_args()

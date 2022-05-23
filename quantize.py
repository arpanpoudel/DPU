import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from models import build_model

DIVIDER = '-----------------------------------------'


def quantize(build_dir, quant_mode, batchsize,args):
    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    # use GPU if available
    if (torch.cuda.device_count() > 0):
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    # load trained model
    model = build_model(args)
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # override batchsize if in test mode
    if (quant_mode == 'test'):
        batchsize = 1

    rand_in = torch.randn([batchsize, 3, 1280, 640])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
    quantized_model = quantizer.quant_model


    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir', type=str, default='build', help='Path to build folder. Default is build')
    ap.add_argument('-q', '--quant_mode', type=str, default='calib', choices=['calib', 'test'],
                    help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b', '--batchsize', type=int, default=100,
                    help='Testing batchsize - must be an integer. Default is 100')
    # * Backbone
    ap.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    ap.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    ap.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    ap.add_argument('--output_dir', default='./logs/',
                        help='path where to save')
    ap.add_argument('--weight_path', default='./weights_poultry/best_mae.pth',
                        help='path where the trained weights saved')

    ap.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--build_dir    : ', args.build_dir)
    print('--quant_mode   : ', args.quant_mode)
    print('--batchsize    : ', args.batchsize)
    print(DIVIDER)

    quantize(args.build_dir, args.quant_mode, args.batchsize,args)

    return


if __name__ == '__main__':
    run_main()
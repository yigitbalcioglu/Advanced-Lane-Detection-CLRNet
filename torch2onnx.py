import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.datasets import build_dataloader
from clrnet.models.registry import build_net


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if 'net' in ckpt_obj and isinstance(ckpt_obj['net'], dict):
            return ckpt_obj['net']
        if 'state_dict' in ckpt_obj and isinstance(ckpt_obj['state_dict'], dict):
            return ckpt_obj['state_dict']
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError('Checkpoint format unsupported: expected dict-like state dict')


def _strip_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    net = build_net(cfg)
    net = net.cpu()
    net.eval()

    ckpt = torch.load(args.load_from, map_location='cpu')
    state_dict = _extract_state_dict(ckpt)
    new_state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = net.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {args.load_from}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    
    dummy_input = torch.randn(1, 3, 320,800, device='cpu')
    # Warm-up once to initialize lazy modules before tracing/export.
    with torch.no_grad():
        _ = net(dummy_input)
    torch.onnx.export(net, dummy_input, args.output,
                    export_params=True, opset_version=16, do_constant_folding=True,
                    input_names = ['input'])
    print(f"ONNX exported: {args.output}")
   
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--load_from',
                    default=None,
                    help='the checkpoint file to load from')
    parser.add_argument('--output', default='model.onnx', help='output onnx path')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()

# python torch2onnx.py configs/clrnet/clr_resnet18_tusimple.py  --load_from tusimple_r18.pth


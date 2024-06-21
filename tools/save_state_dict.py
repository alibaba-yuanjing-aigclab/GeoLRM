import os
import argparse

import torch


def save_state_dict(ckpt_path, save_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    torch.save(state_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('save_path', type=str)
    args = parser.parse_args()
    save_state_dict(args.ckpt_path, args.save_path)

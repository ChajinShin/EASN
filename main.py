import torch
import os
from solver import get_options, get_policy


def main():
    # fetch option and set device
    opt = get_options()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if opt.use_cuda else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx

    policy = get_policy(opt, dev)
    policy.execute()


if __name__ == "__main__":
    main()

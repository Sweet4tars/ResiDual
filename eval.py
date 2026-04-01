import os
import torch
import argparse
import logging
from lib import evaluation


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='f30k', help='dataset name: coco or f30k')
    parser.add_argument('--data_path', type=str, default='data/', help='root path of the dataset metadata')
    parser.add_argument('--model_path', type=str, required=True, help='path to model_best.pth')
    parser.add_argument('--save_results', type=int, default=0, help='whether to save retrieval scores')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info('Evaluating checkpoint: %s', opt.model_path)
    
    model_dir = os.path.dirname(opt.model_path)
    save_path = os.path.join(model_dir, f'results_{opt.dataset}.npy') if opt.save_results else None

    if opt.dataset == 'coco':
        evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='testall', fold5=True)
        evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
    else:
        evaluation.evalrank(opt.model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    
    main()

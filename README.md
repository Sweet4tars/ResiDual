# ResiDual: Residual-based Dual-Granularity Alignment for Cross-Modal Matching

Minimal release repository for training and evaluating **ResiDual** on image-text retrieval benchmarks.

This release only keeps the code and metadata required to run the model:

- training entry: `train.py`
- evaluation entry: `eval.py`
- argument definitions: `arguments.py`
- model code: `lib/`
- caption/id metadata: `data/`

The following research artifacts are intentionally **excluded** from this repository:

- `paper/`
- `runs/`
- `scripts/`
- `visual/`

## Environment

Recommended dependencies:

- Python >= 3.8
- PyTorch >= 1.12
- torchvision >= 0.13
- transformers >= 4.32
- Pillow
- numpy
- tensorboard
- tensorboard_logger

Install with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
ResiDual/
├── arguments.py
├── train.py
├── eval.py
├── lib/
│   ├── encoders.py
│   ├── cross_net.py
│   ├── evaluation.py
│   ├── image_caption.py
│   ├── loss.py
│   ├── utils.py
│   ├── vse.py
│   └── xttn.py
└── data/
    ├── f30k/
    └── coco/
```

## Dataset Preparation

This repository already includes the caption files and image-id mappings for **Flickr30K** and **MS-COCO**.  
You still need to download the raw images and organize them as follows:

```text
data
├── coco
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── dev_ids.txt
│   ├── dev_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
├── f30k
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── dev_ids.txt
│   ├── dev_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
├── flickr30k-images
└── coco-images
    ├── train2014
    └── val2014
```

Default image paths are:

- Flickr30K images: `data/flickr30k-images`
- MS-COCO images: `data/coco-images`

You can override them with:

- `--f30k_img_path`
- `--coco_img_path`

## Pretrained Backbones

ResiDual uses pretrained transformer backbones from Hugging Face:

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- [microsoft/swin-base-patch4-window7-224](https://huggingface.co/microsoft/swin-base-patch4-window7-224)

The models will be downloaded automatically by `transformers` if they are not already cached locally.

## Training

### Single GPU

```bash
# ViT + Flickr30K
python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k_vit \
  --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

# Swin + Flickr30K
python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k_swin \
  --batch_size 64 --vit_type swin --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6

# ViT + COCO
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco_vit \
  --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

# Swin + COCO
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco_swin \
  --batch_size 64 --vit_type swin --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6
```

### Multi-GPU (DDP)

```bash
# ViT + Flickr30K
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py \
  --dataset f30k --multi_gpu 1 --logger_name runs/f30k_vit \
  --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

# Swin + COCO
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
  --dataset coco --multi_gpu 1 --logger_name runs/coco_swin \
  --batch_size 64 --vit_type swin --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6
```

## Evaluation

Evaluation requires an explicit checkpoint path:

```bash
# Flickr30K
python eval.py --dataset f30k --data_path data/ --gpu-id 0 \
  --model_path /path/to/model_best.pth

# MS-COCO
python eval.py --dataset coco --data_path data/ --gpu-id 0 \
  --model_path /path/to/model_best.pth
```

For COCO, the script reports both:

- 5-fold 1K evaluation
- full 5K evaluation

## Checkpoints and Logs

Fill these links manually after upload.

| Setting | Checkpoint | Training Log |
|---|---|---|
| Flickr30K / ViT |  |  |
| Flickr30K / Swin |  |  |
| MS-COCO / ViT |  |  |
| MS-COCO / Swin |  |  |

## Notes

- `runs/` is not versioned in this release. It will be created locally during training.
- The repository only includes the model code and dataset metadata needed to reproduce training/evaluation.
- Paper sources and visualization assets are intentionally omitted from this release package.


# 实验记录

```bash
# One-time preprocess -> cache, then train
python src/train.py --config configs/default.yaml --data_dir /data/yqw/smiles-bd/data/DrugLikeSMILSE-debug --cache_dir /data/yqw/smiles-bd/cache/DrugLikeSMILSE-debug \
--override model.max_len=66 train.batch_size=32 model.d_model=768 

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 src/train.py   --config configs/default.yaml \
   --data_dir /data/yqw/smiles-bd/data/DrugLikeSMILSE-debug \
   --cache_dir /data/yqw/smiles-bd/cache/DrugLikeSMILSE-debug \
   --override model.max_len=66 train.batch_size=100

# Resume
python src/train.py \
  --config configs/default.yaml \
  --resume /data/yqw/smiles-bd/checkpoints/2025-10-24_19-16-51/best_model.pt \
   --data_dir /data/yqw/smiles-bd/data/DrugLikeSMILSE-debug \
   --cache_dir /data/yqw/smiles-bd/cache/DrugLikeSMILSE-debug \
   --override model.max_len=66 train.batch_size=100

# Specific GPU
CUDA_VISIBLE_DEVICES=1,3 \
torchrun --standalone --nproc_per_node=2 src/train.py \
  --config configs/default.yaml

### Sampling
python src/sample.py  --config configs/default.yaml \
  --ckpt /data/yqw/smiles-bd/checkpoints/2025-10-24_16-05-43/best_model.pt \
  --prefix "" --override sample.steps=24 sample.top_p=0.9
```


    data.raw_data_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M \
    data.cache_dir=/share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeSMILES-12B-427M-addBOSEOS \


---


```bash

python -m torch.distributed.run --standalone --nproc_per_node=8 \
   src/train.py   --config configs/default.yaml \
   --data_dir /share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M \
   --cache_dir /share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/smiles-bd-cache-DrugLikeSMILES-12B-427M\
   --override model.max_len=66 train.batch_size=100 model.d_model=1024 model.h_heads=16 model.n_layers=24
```


```bash

python -m torch.distributed.run --standalone --nproc_per_node=8 \
   src/train.py   --config configs/default.yaml \
   --data_dir /share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M \
   --cache_dir /share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeSMILES-12B-427M-addBOSEOS\
   --override model.max_len=66 train.batch_size=100 model.d_model=1024 model.h_heads=16 model.n_layers=24
```

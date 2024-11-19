# terminal commend for running


## One-gpu (we set to use "CUDA_DEVICE = "2" "  in the script)

```
python llama3.2_vision_instruct_train_with_QLora_SFT.py
```

## Multi-gpus 
* use first two gpus
```
python -m torch.distributed.launch --nproc_per_node=2 multigpus_llama3.2_vision_instruct_train_with_QLora_SFT.py
```


* use torchrun, which is the similar function as above  
```
torchrun --nproc_per_node=2 multigpus_llama3.2_vision_instruct_train_with_QLora_SFT.py
```

* with specific gpus ("0" and "1")
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 multigpus_llama3.2_vision_instruct_train_with_QLora_SFT.py
```
## Optimize with deepspeed
* install deepspeed
```
pip install deepspeed
```
* commend run the script with deepspeed config
```
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 multigpus_llama3.2_vision_instruct_train_with_QLora_SFT_deepspeed.py
```
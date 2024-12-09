# Learning progress of large visual language model



# Setup

```
conda create --name ftvlm python=3.10

conda activate ftvlm

pip install "torch==2.4.0" tensorboard pillow

pip install  --upgrade \
  "transformers==4.45.1" \
  "datasets==3.0.1" \
  "accelerate==0.34.2" \
  "evaluate==0.4.3" \
  "bitsandbytes==0.44.0" \
  "trl==0.11.1" \
  "peft==0.13.0" \
  "qwen-vl-utils" \
  "torchvision"

```
# Roadmaps:
-[x] SFT LLAVA-OneVision-7B with (Q)Lora on multi-gpus of deepspeed config

-[x] SFT LLAMA-3.2-11B-Vision-Instruct with (Q)Lora on multi-gpus of deepspeed config

-[x] SFT LLAMA-3.2-11B-Vision-Instruct with (Q)Lora on multi-gpus

-[x] SFT LLAMA-3.2-11B-Vision-Instruct with (Q)Lora

-[x] SFT LLAVA-1.5-7B with (Q)Lora

-[x] SFT LLAVA-1.5-7B
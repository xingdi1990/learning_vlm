#!/usr/bin/env python3
"""
This script implements fine-tuning of the LLAVA OneVision model using QLora and SFT approaches.
"""

import os
import torch
from typing import List, Dict, Any
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from PIL import Image
import io

from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# Configuration
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
# DATASET_ID = "philschmid/amazon-product-descriptions-vlm"
USE_LORA = True
USE_QLORA = False
MULTIPLE_IMAGES_NUM = 2



def setup_environment() -> None:
    """Set up the environment variables and device settings."""

    login("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")

def find_all_linear_names(model: LlavaOnevisionForConditionalGeneration) -> List[str]:
    """Find all linear layer names in the model for LoRA configuration."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model_and_tokenizers():
    """Load and configure the model, processor, and tokenizer."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    if USE_QLORA or USE_LORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # torch.float16
        ) if USE_QLORA else None
        
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, # torch.float16,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, # torch.float16,
        )

    model.config.use_cache = False

    if USE_LORA:

        for param in model.parameters():
            param.requires_grad = True  # Ensure parameters are trainable

        model, config = configure_lora(model)

    return model, processor, config

def configure_lora(model: LlavaOnevisionForConditionalGeneration) -> LlavaOnevisionForConditionalGeneration:
    """Configure LoRA settings for the model."""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config



def prepare_dataset():
    """Load and prepare the dataset for training."""

    def format_data(sample):
        return {
            # "images": [Image.open(io.BytesIO(img)) for img in sample["image"]],
            "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a professional academic paper review assistant."}],
            },
            {   
                "role": "user",
                "content": [
                            # *[{'type': 'image', 'image': Image.open(io.BytesIO(img))} for img in sample["image"]],
                            *[{'type': 'image', 'image': Image.open(io.BytesIO(sample["image"][i])).resize((336, 336))} for i in range(MULTIPLE_IMAGES_NUM)],
                            # {'type': 'image', 'image': Image.open(io.BytesIO(sample["image"][0]))},
                            {"type": "text", "text": "Please help me on reviewing this paper by given those images"}
                            ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["summaries"][0]}],
            },
        ]
        }

    dataset = load_dataset("DetionDX/neurips_openreview_v1", split="train")
    format_dataset = [format_data(sample) for sample in dataset]


    return format_dataset

def get_deepspeed_config():
    return {
        "fp16": {"enabled": False}, #  make sure to match the training_args
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "allgather_partitions": True,
            "reduce_scatter": True
        },
        "gradient_accumulation_steps": "auto", # set auto to match the training_args
        "train_batch_size": "auto", # set auto to match the training_args
        "train_micro_batch_size_per_gpu": "auto" # set auto to match the training_args
        # "gradient_clipping": 0.3,
        # "steps_per_print": 10,
        # "train_micro_batch_size_per_gpu": 1,
        # "gradient_accumulation_steps": 8,  # Set explicitly instead of "auto"
        # "train_batch_size": 16  # micro_batch_size * gradient_accum * num_gpus
    }

def setup_trainer(model, processor, dataset, collate_fn, lora_config=None):
    """Set up the SFT trainer with appropriate configuration."""
    args = SFTConfig(
        output_dir="llava-onevision-qwen2-7b-ov-neurips-openreview-v1",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        torch_compile=False,       # Add this to prevent compilation warnings
        num_train_epochs=10,
        max_seq_length=1024,
        logging_strategy="steps",
        logging_steps=10,
        dataset_text_field="",  # dummy field
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=False, # make sure this is consistant with the model loading and data loading with torch.float16
        bf16=True, # make sure this is consistant with the model loading and data loading with torch.bfloat16
        tf32=True,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.0,                      # warmup ratio based on QLoRA paper
        deepspeed=get_deepspeed_config()  # Add DeepSpeed config
        
    )

    

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=lora_config if lora_config else None, 
        tokenizer=processor.tokenizer,
    )

    return trainer


def main():
    """Main execution function."""
    # Setup
    setup_environment()
    
    # Load model and tokenizers
    model, processor, lora_config = load_model_and_tokenizers()
    
    # Prepare dataset
    formatted_dataset = prepare_dataset()

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        # print(texts)
        # print(examples[0]["messages"])
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        
        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    # check
    print(collate_fn([formatted_dataset[0]]))

    # Setup and start training
    trainer = setup_trainer(model, processor, formatted_dataset, collate_fn, lora_config)
    trainer.train()
    
    # Save the model
    trainer.save_model(trainer.args.output_dir)
    
    # Push to hub if enabled
    if trainer.args.push_to_hub:
        trainer.push_to_hub()
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(trainer.args.hub_model_id)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
LLaMA 3.2 Vision Instruct Training Script with QLora/SFT - Multi-GPU Version
This script implements fine-tuning of the LLaMA 3.2 Vision model using QLora and SFT approaches with distributed training.
"""

import os
import torch
import torch.distributed as dist
from typing import List, Dict, Any
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DATASET_ID = "philschmid/amazon-product-descriptions-vlm"
USE_LORA = True
USE_QLORA = False
NUM_GPUS = torch.cuda.device_count()  # Automatically detect available GPUs

class DataCollator:
    """Data collator for processing text and image pairs."""
    
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(NUM_GPUS)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    


                              
    # Important: Set the device BEFORE initializing process group
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Initialize the distributed environment with explicit device_ids
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=int(os.environ['RANK']),
    )

    # dist.barrier(device_ids=[local_rank])  # Correct way to call barrier

    return device

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def setup_environment() -> None:
    """Set up the environment variables and device settings."""
    os.environ["NCCL_DEBUG"] = "NONE"
    os.environ["NCCL_DEBUG_SUBSYS"] = "NONE"

    login("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")

def find_all_linear_names(model: MllamaForConditionalGeneration) -> List[str]:
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

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model_and_tokenizers(local_rank):
    """Load and configure the model, processor, and tokenizer."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if USE_QLORA or USE_LORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ) if USE_QLORA else None
        
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map={'': local_rank}
        )
    else:
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={'': local_rank}
        )

    if USE_LORA:
        model = configure_lora(model)

    return model, processor, tokenizer

def configure_lora(model: MllamaForConditionalGeneration) -> MllamaForConditionalGeneration:
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
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model

def prepare_dataset(processor):
    """Load and prepare the dataset for training."""
    dataset = load_dataset(DATASET_ID, split="train")
    
    prompt_template = """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""

    def format_data(sample):
        return {
            "messages": [
                {'content': [
                    {'text': prompt_template.format(
                        product_name=sample["Product Name"],
                        category=sample["Category"]
                    ), 'type': 'text'},
                    {'text': None, 'type': 'image'}
                ], 'role': 'user'},
                {'content': [{'text': sample["description"], 'type': 'text'}],
                'role': 'assistant'},
            ],
            "images": [sample["image"]],
        }

    formatted_dataset = [format_data(sample) for sample in dataset]
    return formatted_dataset

def setup_trainer(model, processor, formatted_dataset, local_rank):
    """Set up the SFT trainer with appropriate configuration."""
    training_args = SFTConfig(
        output_dir="./llama3.2_vision_instruct_output",
        learning_rate=1.4e-5,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        max_seq_length=1024,
        dataset_text_field="text",  # dummy field
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        local_rank=local_rank,
        deepspeed=None,  # Add DeepSpeed config here if needed
        ddp_backend="nccl",
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=500,
        eval_strategy="no",
    )

    data_collator = DataCollator(processor)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        eval_dataset=None,
        tokenizer=processor.tokenizer,
        data_collator=data_collator
    )

    return trainer

def main():
    """Main execution function."""
    # Get local rank from environment
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Setup distributed training and get device
    device = setup_distributed()
    setup_environment()
    
    # Load model and move to correct device
    model, processor, tokenizer = load_model_and_tokenizers(local_rank)
    model = model.to(device)
    
    # Rest of your training code remains the same
    formatted_dataset = prepare_dataset(processor)
    trainer = setup_trainer(model, processor, formatted_dataset, local_rank)
    trainer.train()
    
    if dist.get_rank() == 0:
        trainer.save_model(trainer.args.output_dir)
        if trainer.args.push_to_hub:
            trainer.push_to_hub()
            processor.push_to_hub(trainer.args.hub_model_id)
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
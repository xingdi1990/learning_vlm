
def main():
    #!/usr/bin/env python
    # coding: utf-8
    
    # # Set up the environment
    # https://huggingface.co/docs/transformers/en/model_doc/llava
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    
    
    # # Load the model
    
    
    
    # Unlike direct load the pretrained model, we choose the PEFT strategy for finetuning with Lora.
    # https://huggingface.co/docs/peft/en/index
    
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    USE_LORA = True
    USE_QLORA = False
    
    ## Load model
    
    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # torch.float16,
        )
    
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    if USE_LORA:
    
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
    
    
    # # Load the dataset
    from datasets import load_dataset
    
    dataset_id = "philschmid/amazon-product-descriptions-vlm"
    dataset = load_dataset(dataset_id, split="train")
    print(dataset)
    
    
    # # Display an example from the dataset
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    
    def display_example(example):
        print(f"Product Name: {example['Product Name']}")
        print(f"Category: {example['Category']}")
        print(f"Description: {example['description']}")
        print("image:")
        
        # Convert bytes to PIL Image
        image = example['image']
    
        # Display using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show()
    
    # Display an example
    display_example(dataset[0])
    
    
    # # Format the dataset for the TRL trainer
    #  https://huggingface.co/docs/trl/en/sft_trainer
    
    prompt = """
    
    ##PRODUCT NAME##: {product_name}
    ##CATEGORY##: {category}
    
    """
    
    
    def format_data(sample):
        return {    
            "images": [sample["image"]],
            "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image. Only return description. The description should be SEO optimized and for a better mobile search experience."}],
            },
            {   
                "role": "user",
                "content": [{'type': 'image', "text": None },
                            {"type": "text", "text": prompt.format(product_name=sample["Product Name"], category=sample["Category"])}
                            ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ]
        }
    
    
    train_valid = dataset.train_test_split(test_size=0.2)
    train_dataset = train_valid["train"]
    valid_dataset = train_valid["test"]
    
    
    formatted_train_ds = [format_data(sample) for sample in train_dataset]
    formatted_val_ds = [format_data(sample) for sample in valid_dataset]
    print(formatted_train_ds[0])
    print(formatted_val_ds[0])
    
    
    # # Create a data collator to encode text and image pairs
    class DataCollator:
        def __init__(self, processor):
            self.processor = processor
        ################
        # Create a data collator to encode text and image pairs
        ################
        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                messages = example["messages"]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(example["images"][0])
    
            batch = self.processor(texts, images, return_tensors="pt", padding=True)
    
            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
    
            return batch
        
    
    data_collator = DataCollator(processor)
    
    # print(processor.apply_chat_template(formatted_train_ds[0]["messages"], tokenize=False, add_generation_prompt=False))
    print(data_collator([formatted_train_ds[0]]))
    print(data_collator([formatted_val_ds[0]]))
    
    
    
    
    from trl import (
        ModelConfig,
        SFTConfig,
        SFTTrainer,
        TrlParser,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )
    
    training_args = SFTConfig(
        output_dir="llava-1.5-7b-hf-philschmid-amazon-product-descriptions-vlm",
        learning_rate= 2e-4,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        max_seq_length=1024,
        dataset_text_field="text",  # need a dummy field
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=False, # make sure this is consistant with the model loading and data loading with torch.float16
        bf16=True, # make sure this is consistant with the model loading and data loading with torch.bfloat16
        tf32=True,
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_ds,
        eval_dataset=formatted_val_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    

if __name__ == "__main__":
    main()

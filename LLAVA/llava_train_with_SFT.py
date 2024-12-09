
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.bfloat16 # torch.float16 
    )
    
    
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

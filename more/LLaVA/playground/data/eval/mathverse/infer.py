import argparse
import json
import math
import os
import re
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
# from gllava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
# from gllava.utils import disable_torch_init
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,IMAGE_PLACEHOLDER

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import requests
from io import BytesIO
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

prompt_choice = {
    "none": "",
    "single": "Answer the question using a single word or phrase.",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_boxed_content(text):
    pattern = r'\\boxed\{(.*?)\}'
    match_list = re.findall(pattern, text)
    if match_list == []:
        return None
    else:
        return match_list[-1]


def evalmodel(args, model_name, tokenizer, model, image_processor, context_len):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    # print(image_files[0])
    images = load_images(image_files)
    # print(images[0])
    image_sizes = [x.size for x in images]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def eval_model(args):
    # Model
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_path,
    #     model_base=None,
    #     model_name=model_name,
    #     num_of_kvs=args.num_of_kvs,
    #     merge_version=args.merge_version
    # )

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
    )


    disable_torch_init()

    # Data
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answers_file_dir = os.path.dirname(answers_file)
    os.makedirs(answers_file_dir, exist_ok=True)
    answers_file = os.path.join(answers_file_dir, f"{args.chunk_idx}_{os.path.basename(answers_file)}")
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['query_cot']
        query = f"{question}\n"
        query += prompt_choice[args.prompt]
        if line['image'] == '':
            image_path = "/mnt/bn/pengshuai-nas/MathLLM/LLaVA/fake_image_336.png"
        else:
            image_path = os.path.join(args.image_folder, line["image"])

        args_llava = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": query,
            "conv_mode": args.conv_mode,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024
        })()




        outputs = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)

        # print(outputs)




        if args.prompt == 'multimath':
            response = extract_boxed_content(outputs)
        else:
            response = outputs
        
        line['model_output'] = outputs
        line['response'] = response
        ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image_folder", type=str, default="/")
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    parser.add_argument("--prompt", type=str, choices=['none', 'single', 'multimath'])
    parser.add_argument("--conv_mode", type=str, default="dpsk")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_of_kvs", type=int, default=4)
    parser.add_argument("--merge_version", type=str, default='cross_channel')

    args = parser.parse_args()

    eval_model(args)

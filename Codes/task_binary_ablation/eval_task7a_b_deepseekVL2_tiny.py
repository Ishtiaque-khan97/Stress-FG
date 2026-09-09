# ######### used this part for fixing problems running on ARC #

import os


os.environ['HF_HOME'] = '/projects/abbott_lab/Users/ishtiaque/hfmodels/'
os.environ['HF_HUB_CACHE'] = '/projects/abbott_lab/Users/ishtiaque/hfmodels/'
os.environ['XDG_CACHE_HOME'] = '/projects/abbott_lab/Users/ishtiaque/hfmodels/'
os.environ['NB_USER'] = 'ishtiahmed'#'ishtiaqueahmedk'
os.environ['TRANSFORMERS_CACHE'] = '/projects/abbott_lab/Users/ishtiaque/hfmodels/'
os.environ['HF_DATASETS_CACHE'] = '/projects/abbott_lab/Users/ishtiaque/hfmodels/'


# normal general imports

import os
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
import os 
import json
import random
from tqdm import tqdm
from collections import Counter 

import numpy as np
import torch


import json
import base64
import os
import re
from pathlib import Path
import time


from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random



from datetime import datetime

from general_code import *


################## ################## ################## 
# Model SPecific Code 
# ###################### ################## ################## 


output_file_name = "deepseekVL2_tiny_8B_task7a_b_results_log_medium.txt"


import sys
sys.path.append('/projects/abbott_lab/Users/ishtiaque/hfmodels/DeepSeek-VL2')

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def get_answer(image_path, query):

    full_query = f"<image>\n{query}"


    # print(query)
    
    conversation = [
        {
            "role": "<|User|>",
            # "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
            "content": full_query,
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)
    
    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    # print(f"{prepare_inputs['sft_format'][0]}", answer)
    answer2 = answer.replace("<｜end▁of▁sentence｜>", "")


    
    
    return answer2

################## ################## ################## 
# General Code 
# ###################### ################## ################## 





# json_files_list = ['negated_questions', 'new_cub_class_descriptions','new_cub_with_class_descriptions', 'modified_new_cub_class_descriptions_full_fake', 'modified_new_cub_class_descriptions_partial_fake', 'incorrect_class_correct_description', 'correct_class_incorrect_description']

bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

bird_list = [
    "new_cub_class_descriptions",
    "new_cub_class_descriptions_task_0b_class_only_baseline",
]

food_list = [
    "new_food_class_descriptions",
    "new_food_class_descriptions_task_0b_class_only_baseline",
]

aircraft_list = [
    "new_aircraft_class_descriptions",
    "new_aircraft_class_descriptions_task_0b_class_only_baseline",
]

dogs_list = [
    "new_dogs_class_descriptions",
    "new_dogs_class_descriptions_task_0b_class_only_baseline",
]

car_list = [
    "new_car_class_descriptions",
    "new_car_class_descriptions_task_0b_class_only_baseline",
]


# json_files_list = food_list  #################### change!!!!!!!!!!!!!!!!!!!!
all_lists = [bird_list, food_list, aircraft_list, dogs_list, car_list]
folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]



for json_files_list, images_folder in zip(all_lists, folder_lists):

    bird_images = get_bird_images(images_folder)
    
    for json_file_name in json_files_list:
        
        json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file_name}.json"
        json_data = get_json_data(json_file_name)
    
        # json_data[0]
        # json_data[1]
    
        medium_data, hard_data = get_medium_hard_data(json_data)
        
        print("\n----Medium----")
        with open(output_file_name, "a", encoding="utf-8") as f:
            print(f"\n ------- Medium ------- ", file=f)

        run_eval(medium_data, bird_images, get_answer, json_file_name, output_file_name)

        # print("\n----Hard----")
        # with open(output_file_name, "a", encoding="utf-8") as f:
        #     print(f"\n ------- Hard ------- ", file=f)
        # # run_eval(hard_data)
        # run_eval(hard_data, bird_images, get_answer, json_file_name, output_file_name)
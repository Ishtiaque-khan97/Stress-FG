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


output_file_name = "llava_ov_8B_task7a_b_results_log_csv_sim.txt"


from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM


model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def get_answer(image_path, query):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": query},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]
    
    
    

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

        print("\n----Hard----")
        with open(output_file_name, "a", encoding="utf-8") as f:
            print(f"\n ------- Hard ------- ", file=f)
        run_eval(hard_data, bird_images, get_answer, json_file_name, output_file_name)
    
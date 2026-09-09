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





# ======================================================================

# model-specific

# model-specific imports
# Example usage:
output_file_name = "internvl_task4_results_log.txt"


import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import accelerate 



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12): #448
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values




# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)



generation_config = dict(
    max_new_tokens=64,  # Only need a few tokens for single letter #10
    do_sample=False,    # Deterministic output
    temperature=0.0     # No randomness
)


def get_answer(answ_img_paths, query):

    query = f"<image><image><image><image><image>\n{query}"
    
    pixel_values_list = []
    for image_path in answ_img_paths:

        pixel_val = load_image(image_path, max_num=8).to(torch.bfloat16).cuda() #max_num=12
        pixel_values_list.append(pixel_val)
    
    pixel_values = torch.cat(pixel_values_list, dim=0)
                
    output_text = model.chat(tokenizer, pixel_values, query, generation_config, return_history=False)
    
    return output_text
    

# ============================================= general stuff


def get_bird_images(images_folder):
# Path to the folder containing bird subfolders
    # images_folder = '/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images'
    
    # Dictionary to store bird names and their corresponding image paths
    bird_images = {}
    
    # Iterate over each subfolder in the images folder
    for folder in os.listdir(images_folder):
        bird_name = folder.split(".")[-1]  # Extract bird name from folder name
        folder_path = os.path.join(images_folder, folder)  # Path to the bird's folder
        
        # Initialize an empty list to store image paths for the current bird
        image_paths = []
        
        # Iterate over the image files in the bird's folder
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)  # Full path to the image file
            image_paths.append(image_path)  # Store the image path
        
        # Store the list of image paths in the dictionary under the bird's name
        bird_images[bird_name] = image_paths
    
    # Now bird_images contains a dictionary where the keys are bird names and the values are lists of image paths
    print(f"len(bird_images): {len(bird_images)}")
    return bird_images


def get_json_data(json_file_name):

    # negated_questions, modified_new_cub_class_descriptions_full_fake, #modified_new_cub_class_descriptionsx #modified_mcqs_description_only2, modified_mcqs_description_only

    # For Task 1 type 1: 
    with open(json_file_name, "r") as file: 
        json_data = json.load(file)
    print(f"len(json_data): {len(json_data)}")
    return json_data


def get_map_dict(json_data):
    
    #create dictionary ishti

    captions_mcqid_pair_dict = {}
    class_freq_dict = {}
    
    for mcq in json_data:
        
        options = mcq['options'] # get the answer choices dictionary of the first mcq
        correct_option_description = {options[mcq['correct_answer']]} # get the correct answer caption
        class_name = mcq['mcq_id']
    
        # make sure that there is only one correct caption
        assert len(correct_option_description)==1, "More than one correct answer!!"
    
        #convert to string
        correct_option_description = next(iter(correct_option_description))
    
        if correct_option_description in captions_mcqid_pair_dict: # if caption already in dict
            
            # get the existing mcq_id and make sure it matches
            exist_class_name = captions_mcqid_pair_dict[correct_option_description] 
            
            # make sure mcq_id matches
            assert exist_class_name == class_name, f"mismatch in class name: [{exist_class_name}] and [{class_name}]"
            class_freq_dict [class_name] = class_freq_dict [class_name] + 1
            
    
                
        else:
            captions_mcqid_pair_dict[correct_option_description] = class_name
    
            class_freq_dict [class_name] = 1 

    return captions_mcqid_pair_dict
    
    
    # print(f"Successfully created class_names dictionary with {len(captions_mcqid_pair_dict)} Class entries from JSON file:\n {filename}.\n")
    
    # print(captions_mcqid_pair_dict)
    # print(class_freq_dict)

    
def get_medium_hard_data(json_data):
    
# Use this if the json file contain the medium and hard categories
    medium_data = []
    hard_data = []

    data_counter = 0
    use_partial = False#True#False
    if use_partial:
        print("using limited data for debugging")
    
    for data in json_data:
        if data['difficulty'] == "Medium":
            medium_data.append(data)
        else:
            hard_data.append(data)        

        data_counter = data_counter + 1
        if (data_counter>120) and use_partial:
            print(f"stopping at data = {data_counter}")
            break
        
    print(len(medium_data))
    print(len(hard_data))
    return medium_data, hard_data
    
    

def run_eval(data_partition):
    

    # Counters for distribution
    true_distribution = Counter()
    predicted_distribution = Counter()
    
    results = []
    
    for i, item in (enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        # print (f"mcq_id is: {mcq_id}\n")
        # question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        modif_correct_answer = item['modif_correct_answer']
        modification_caption = item['modification_caption']
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:5]


        # # Format the prompt
        # #original
        # formatted_prompt = f"I am giving a reference image (first Image) of a reference bird. I am looking for a different bird. {modification_caption}. Which of the following four new images (A, B, C or D) apart from the reference image matches best with this description?\n"
            
    
        # # Final prompt
        # prompt = f""" Your answer or response must ONLY be a single index ('A', 'B', 'C', 'D'). Do not response with any other text. 
    
        # {formatted_prompt}
    
        # Answer: ('A', 'B', 'C', 'D')"""

        # # Format the prompt
        # #ishti modified for E
        # formatted_prompt = (
        #     "You are given five images labeled A, B, C, D, and E.\n"
        #     "Image A is a reference image of a bird.\n"
        #     "You are looking for a different bird that matches the following description:\n"
        #     f"{modification_caption}\n\n"
        #     "Which ONE of images B, C, D, or E best matches this description?"
        # )
        
        # prompt = (
        #     "Output exactly ONE character: B, C, D, or E.\n"
        #     "Do not output any other text.\n\n"
        #     f"{formatted_prompt}\n\n"
        #     "Answer:"
        # )


        # # Format the prompt
        # #chatgpt modified for  E
    
        prompt = (
            "You are given five images labeled A, B, C, D, and E.\n"
            "Image A (Image 1) is a reference image of a reference bird.\n"
            "Images B, C, D, and E show different birds.\n\n"
            "Find the bird that matches the following description:\n"
            f"{modification_caption}\n\n"
            "Which ONE of images B, C, D, or E best matches the description?\n\n"
            "Output exactly ONE character: B, C, D, or E.\n"
            "Do not output any other text.\n\n"
            "Answer:"
        )



        # Run the model
        for i, image_path in enumerate(image_paths):
    
            #get the image paths for the reference image and the four answer options
            answ_img_paths = []
            answ_img_paths.append(image_path) # ref image
            ref_class_option_count = (i+1)%5

            
            for k in ['A', 'B', 'C', 'D']:  # ['D', 'C', 'B', 'A'] for position bias checking ['A', 'B', 'C', 'D']
                # formatted_prompt += f"{k}. {options[k]}\n"
                
                if (correct_answer == k):
                    answ_img_paths.append(image_paths[ref_class_option_count]) # get an image for the mcq option showing the reference class.
                    continue
                description = options[k]
                answ_mcq_id = captions_mcqid_pair_dict[description]
                answ_mcq_img_path = bird_images[answ_mcq_id][0]
                answ_img_paths.append(answ_mcq_img_path)
                
            
    
                
            model_output = get_answer(answ_img_paths, prompt)
            # print(f"Model Output: {model_output}, Correct Answer: {modif_correct_answer}", )
            
    
            # # # Extract predicted answer (basic string search, can refine)
            # # # originl code
            # predicted_answer = None
            # for option in ['A', 'B', 'C', 'D']:
            #     if f"{option}" in model_output or f"{option}." in model_output:
            #         predicted_answer = option


            # Extract predicted answer (basic string search, can refine)
            # get previous letter code
            orig_options_list = ['A', 'B', 'C', 'D']
            modif_options_list = ['B', 'C', 'D', 'E']
            predicted_answer = None
            for op_i, option in enumerate (modif_options_list):
                if f"{option}" in model_output or f"{option}." in model_output:
                    predicted_answer = orig_options_list[op_i]
                    break
                    

            if predicted_answer is None:
                print(f"model_output: {model_output}")
                print("Error in model_output!!!!!!!!")
                predicted_answer = 'X'
                # print(asd)

            
            # Update counters
            # true_distribution[modif_correct_answer] += 1
            # if predicted_answer:
            #     predicted_distribution[predicted_answer] += 1
            
            results.append({
                'mcq_id': mcq_id,
                'image_path': image_path,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_answer': predicted_answer,
                'correct_answer': modif_correct_answer,
                'is_correct': predicted_answer == modif_correct_answer
            })
    
    # Accuracy summary 
    print(f"Results for file: {json_file_name}")
    
    correct = sum(r['is_correct'] for r in results if r['predicted_answer'] is not None)
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}") 
    
    # Print distributions
    # print("True Option Distribution:", dict(true_distribution))
    # print("Predicted Option Distribution:", dict(predicted_distribution))
    
    # Open the file in "a" (append) mode. 
    # Using 'with' ensures the file is safely closed after writing.

    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Results for file: {json_file_name}", file=f)
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}", file=f)
        # print("True Option Distribution:", dict(true_distribution), file=f)
        # print("Predicted Option Distribution:", dict(predicted_distribution), file=f)

    return prompt, answ_img_paths
    
    
    

bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

# folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]

# json_files_list = ["task_4_composed_questions_bird_fixed", "task_4_composed_questions_food_fixed", "task_4_composed_questions_aircraft_fixed", "task_4_composed_questions_dogs_fixed", "task_4_composed_questions_car_fixed"]

folder_lists = [ car_folder]

json_files_list = ["task_4_composed_questions_car_fixed"]


from datetime import datetime

for json_file, images_folder in zip(json_files_list, folder_lists):

    bird_images = get_bird_images(images_folder)
    json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file}.json"

    json_data = get_json_data(json_file_name)
    captions_mcqid_pair_dict = get_map_dict(json_data)

    medium_data, hard_data = get_medium_hard_data(json_data)

    
    
    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Medium----", file=f)
    
    prompt, answ_img_paths = run_eval(medium_data)

    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Hard----", file=f)
    prompt, answ_img_paths = run_eval(hard_data)

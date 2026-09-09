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
import torchvision.transforms as T

import re
import os
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


from datetime import datetime

import base64
import time

######### 


# model-specific

# model-specific imports

output_file_name = "qwen3_6_27B_task1_image_results_log.txt"

from openai import OpenAI
import argparse

print("Wait starting...")
time.sleep(40)  # Pauses for 10 seconds
print("40 seconds have passed.")

openai_api_key = "xxxx"
# openai_api_base = "http://localhost:8082/v1" 35B - A3B
openai_api_base = "http://localhost:8084/v1" # 27B

client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

models = client.models.list()
model = models.data[0].id
print(f"============\nModel is === {model}\n============")




import base64

def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file content to base64 format."""

    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")

    return result


import re


def get_answer(answ_img_paths, query):

    query = query + " The given images are mapped from A to D."

    encoded_images = [
        encode_base64_content_from_file(img_path)
        for img_path in answ_img_paths
    ]

    content = []
    for encoded_image in encoded_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            },
        })

    content.append({"type": "text", "text": query})

    chat_completion_from_local_image_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=model,
        max_tokens=32,
        temperature=0.1,
        top_p=1,
        # presence_penalty=1.5,
        extra_body={
            # "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    result = chat_completion_from_local_image_base64.choices[0].message.content
    if result is None:
        return "X"

    match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
    final_answer = match.group(1).strip() if match else result

    return final_answer




##################
# general code

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
    print(len(bird_images))
    return bird_images


def get_json_data(json_file_name):

    # negated_questions, modified_new_cub_class_descriptions_full_fake, #modified_new_cub_class_descriptionsx #modified_mcqs_description_only2, modified_mcqs_description_only

    # For Task 1 type 1: 
    with open(json_file_name, "r") as file: 
        json_data = json.load(file)
    print(len(json_data))
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
        if (data_counter>50) and use_partial:
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
    bad_ans = 0
    
    for i, item in tqdm(enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:5]
    
        # Format the prompt
        description = options[correct_answer]
        formatted_prompt = f"Which image (A, B, C or D) matches best with this description: {description}?\n"
    
    
        
    
        # Final prompt
        prompt = f""" Your answer or response must ONLY be a single index ('A', 'B', 'C', 'D'). Do not response with any other text. 
    
        {formatted_prompt}
    
        Answer: ('A', 'B', 'C', 'D')"""
    
        # Run the model
        for image_path in image_paths:
    
            #get the image paths for the four answer options
            answ_img_paths = []
            for k in ['A', 'B', 'C', 'D']:  # ['D', 'C', 'B', 'A'] for position bias checking ['A', 'B', 'C', 'D']
                # formatted_prompt += f"{k}. {options[k]}\n"
                
                if (correct_answer == k):
                    answ_img_paths.append(image_path)
                    continue
                description = options[k]
                answ_mcq_id = captions_mcqid_pair_dict[description]
                answ_mcq_img_path = bird_images[answ_mcq_id][0]
                answ_img_paths.append(answ_mcq_img_path)
                
            # print(answ_img_paths)
            
    
                
            model_output = get_answer(answ_img_paths, prompt)
            # print("Model Output: ", model_output)
    
            # Extract predicted answer (basic string search, can refine)
            predicted_answer = None
            found_at_least_one = 0
            for option in ['A', 'B', 'C', 'D']:
                if f"{option}" in model_output or f"{option}." in model_output:
                    predicted_answer = option
                    found_at_least_one = 1
                    break
    
            # Update counters
            true_distribution[correct_answer] += 1
            if predicted_answer:
                predicted_distribution[predicted_answer] += 1

            if found_at_least_one == 0:
                bad_ans = bad_ans + 1
                print (f"Model not following output format!! {model_output}")

                
            results.append({
                'mcq_id': mcq_id,
                'image_path': image_path,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'is_correct': predicted_answer == correct_answer
            })
    
    # Accuracy summary 
    print(f"Results for file: {json_file_name}")
    
    correct = sum(r['is_correct'] for r in results if r['predicted_answer'] is not None)
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}") 
    
    # Print distributions
    print("True Option Distribution:", dict(true_distribution))
    print("Predicted Option Distribution:", dict(predicted_distribution))

    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Results for file: {json_file_name}", file=f)
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}", file=f)
        print(f"bad_ans: {bad_ans}/{total} = {bad_ans / total:.2%}", file=f)
    
    return prompt, answ_img_paths



# run-3
# json_files_list = ['negated_questions', 'new_cub_class_descriptions','new_cub_with_class_descriptions', 'modified_new_cub_class_descriptions_full_fake', 'modified_new_cub_class_descriptions_partial_fake', 'incorrect_class_correct_description', 'correct_class_incorrect_description']

bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

bird_list = ["new_cub_class_descriptions", "new_cub_class_descriptions_task_0a_with_class_baseline"]
food_list = ["new_food_class_descriptions", "new_food_class_descriptions_task_0a_with_class_baseline"]
aircraft_list = ["new_aircraft_class_descriptions", "new_aircraft_class_descriptions_task_0a_with_class_baseline"]
dogs_list = ["new_dogs_class_descriptions", "new_dogs_class_descriptions_task_0a_with_class_baseline"]
car_list = ["new_car_class_descriptions", "new_car_class_descriptions_task_0a_with_class_baseline"]


# json_files_list = food_list  #################### change!!!!!!!!!!!!!!!!!!!!
all_lists = [bird_list, food_list, aircraft_list, dogs_list, car_list]
folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]

for json_files_list, images_folder in zip(all_lists, folder_lists):

    print(f"Number of JSON files: {len(json_files_list)}\n")

    bird_images = get_bird_images(images_folder)
    
    for json_file_name in json_files_list:
        
        json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file_name}.json"
        json_data = get_json_data(json_file_name)
        
        medium_data, hard_data = get_medium_hard_data(json_data)

        captions_mcqid_pair_dict = get_map_dict(json_data)
        
        # print("\n----Medium----")
        # prompt, answ_img_paths = run_eval(medium_data)
        print("\n----Hard----")
        with open(output_file_name, "a", encoding="utf-8") as f:
            print(f"\n ------- Hard ------- ", file=f)
        prompt, answ_img_paths = run_eval(hard_data)
    
    
    

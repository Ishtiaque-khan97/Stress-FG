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

output_file_name = "scout_api_task3_mitigate_results_log_new.txt"


from openai import OpenAI
import argparse

'''
 ssh -f -N -L 8082:fal042:43979 ishtiaqueahmedk@falcon1.arc.vt.edu  
'''


# dedicated gpu server through interactive apps
# Modify OpenAI's API key and API base to use the server.
openai_api_key = "xxxxxx"
openai_api_base = "http://localhost:8082/v1"


client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

models = client.models.list()
model = models.data[0].id
# print(models.data[0].id)
print(model)

import base64

def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file content to base64 format."""

    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")

    return result


import re


def get_answer(image_path, query):

    local_image_base64 = encode_base64_content_from_file(image_path)
    chat_completion_from_local_image_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{local_image_base64}"
                        },
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=32,
    )

    result = chat_completion_from_local_image_base64.choices[0].message.content
    # print("Chat completion output from base64 encoded local image:", result)


    match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
    final_answer = match.group(1).strip() if match else result

    # print(result)
    # print(final_answer)
    return final_answer
    
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
    


# # Image-only ---------------------------------------------------------------------------


def run_eval_image_only(data_partition):

    with open(output_file_name, "a", encoding="utf-8") as f:
        print (f"\n----------Running in Image-only mode----------\n", file=f)

    
    
    # Counters for distribution
    true_distribution = Counter()
    predicted_distribution = Counter()
    
    results = []
    bad_answers = 0
    
    for i, item in tqdm(enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        correct_description =  item["correct_description"]
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:5]
    
        # Format the prompt
        formatted_prompt = f"{question}\n"

        
        for k in ['A', 'B', 'C', 'D']:  # ['D', 'C', 'B', 'A'] for position bias checking ['A', 'B', 'C', 'D']
            formatted_prompt += f"{k}. {options[k]}\n"
    
        # Final prompt
        # prompt = f""" First, generate a description of the object in the image. Then, use this for helping to answer the following question. Do not show your reasoning or explanation. Your final answer or response must be a in this format 'Final: A' or 'Final: B' etc. Make sure this is the format of the final answer. 
    
        # {formatted_prompt}
    
        # Answer options are: ('A', 'B', 'C', 'D')"""

        # prompt with correct description caption added.
        prompt = f""" Please answer the following question. Your final answer or response must be in this format 'Final: A' or 'Final: B' etc.
    
        {formatted_prompt}
    
        Do NOT show any reasoning. Answer options are: ('A', 'B', 'C', 'D')"""


        #         # prompt without correct description caption .
        # prompt = f""" Your answer or response must ONLY be a single index ('A', 'B', 'C', 'D'). Do not response with any other text. 
    
        # {formatted_prompt}
    
        # Answer: ('A', 'B', 'C', 'D')"""

        

    
        # Run the model
        for image_path in image_paths:
            model_output = get_answer(image_path, prompt)
            model_output = model_output.replace(" ", "")
            # print(f"Model Output:{model_output}")
            
    
            # Extract predicted answer (basic string search, can refine)
            predicted_answer = None
            found_at_least_one = 0
            for option in ['A', 'B', 'C', 'D']:
                # if f"{option}" in model_output or f"{option}." in model_output:#orig
                if f"Final:{option}" in model_output or f"Final:{option}." in model_output or f"Final:'{option}'." in model_output or f"'{option}'" in model_output:
                    predicted_answer = option
                    found_at_least_one = 1
                    
                elif len(model_output.strip())<5 and (f"{option}" in model_output or f"{option}." in model_output):
                    predicted_answer = option
                    found_at_least_one = 1
                    
            if found_at_least_one == 0:
                print (f"model_output: {model_output}")
                print("Model output not following answer option format")
                predicted_answer = "X"
                bad_answers = bad_answers + 1
                # raise ValueError("Model output not following answer option format")


            # Update counters
            true_distribution[correct_answer] += 1
            if predicted_answer:
                predicted_distribution[predicted_answer] += 1
            
            results.append({
                'mcq_id': mcq_id,
                'image_path': image_path,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'is_correct': predicted_answer == correct_answer
            })
        # if i>10:
        #     break
    
    print(f"Results for file: {json_file_name}")
    # Accuracy summary 
    correct = sum(r['is_correct'] for r in results if r['predicted_answer'] is not None)
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}") 
    
    # Print distributions
    print("True Option Distribution:", dict(true_distribution))
    print("Predicted Option Distribution:", dict(predicted_distribution))   
    
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Image-only Negation Results for file: {json_file_name}", file=f)
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}", file=f)
        print(f"Bad Answers: {bad_answers}/{total} = {bad_answers / total:.2%}", file=f)

    
    


bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]
json_files_list = ["task_3_negated_questions_bird_merged_fixed", "task_3_negated_questions_food_merged_fixed", "task_3_negated_questions_aircraft_merged_fixed", "task_3_negated_questions_dog_merged_fixed", "task_3_negated_questions_car_merged_fixed"]

from datetime import datetime

for json_file, images_folder in zip(json_files_list, folder_lists):

    bird_images = get_bird_images(images_folder)
    json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file}.json"

    json_data = get_json_data(json_file_name)
    # captions_mcqid_pair_dict = get_map_dict(json_data)

    medium_data, hard_data = get_medium_hard_data(json_data)

    
    
    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Medium----", file=f)
    
    run_eval_image_only(medium_data)

    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Hard----", file=f)
    run_eval_image_only(hard_data)



# # Image + Correct Description Caption




def run_eval_image_text(data_partition):


    with open(output_file_name, "a", encoding="utf-8") as f:
        print (f"\n----------Running in Image+Text mode----------\n", file=f)
    
    
    
    # Counters for distribution
    true_distribution = Counter()
    predicted_distribution = Counter()
    
    results = []
    bad_answers = 0
    
    for i, item in tqdm(enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        correct_description =  item["correct_description"]
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:5]
    
        # Format the prompt
        formatted_prompt = f"{question}\n"

        
        for k in ['A', 'B', 'C', 'D']:  # ['D', 'C', 'B', 'A'] for position bias checking ['A', 'B', 'C', 'D']
            formatted_prompt += f"{k}. {options[k]}\n"
    
        # prompt with correct description caption added.
        prompt = f""" Here is a description of the target object in an image: "{correct_description}"\n. Please answer the following question. Your final answer or response must be in this format 'Final: A' or 'Final: B' etc.
    
        {formatted_prompt}
    
        Do NOT show any reasoning. Answer options are: ('A', 'B', 'C', 'D')"""


        #         # prompt without correct description caption .
        # prompt = f""" Your answer or response must ONLY be a single index ('A', 'B', 'C', 'D'). Do not response with any other text. 
    
        # {formatted_prompt}
    
        # Answer: ('A', 'B', 'C', 'D')"""

        

    
        # Run the model
        for image_path in image_paths:
            model_output = get_answer(image_path, prompt)
            # print(f"Model Output:{model_output}")
            model_output = model_output.replace(" ", "")
            
    
            # Extract predicted answer (basic string search, can refine)
            predicted_answer = None
            found_at_least_one = 0
            for option in ['A', 'B', 'C', 'D']:
                # if f"{option}" in model_output or f"{option}." in model_output:#orig
                if f"Final:{option}" in model_output or f"Final:{option}." in model_output or f"Final:'{option}'." in model_output or f"'{option}'" in model_output:
                    predicted_answer = option
                    found_at_least_one = 1
                    
                elif len(model_output.strip())<5 and (f"{option}" in model_output or f"{option}." in model_output):
                    predicted_answer = option
                    found_at_least_one = 1
                    
            if found_at_least_one == 0:
                # print (f"model_output: {model_output}")
                # print("Model output not following answer option format")
                predicted_answer = "X"
                bad_answers = bad_answers + 1
                # raise ValueError("Model output not following answer option format")            
    


            # Update counters
            true_distribution[correct_answer] += 1
            if predicted_answer:
                predicted_distribution[predicted_answer] += 1
            
            results.append({
                'mcq_id': mcq_id,
                'image_path': image_path,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'is_correct': predicted_answer == correct_answer
            })
        # if i>10:
        #     break
    
    print(f"Results for file: {json_file_name}")
    # Accuracy summary 
    correct = sum(r['is_correct'] for r in results if r['predicted_answer'] is not None)
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}") 
    
    # Print distributions
    print("True Option Distribution:", dict(true_distribution))
    print("Predicted Option Distribution:", dict(predicted_distribution))   
    
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Negation: Image with correct description Results for file: {json_file_name}", file=f)
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}", file=f)
        print(f"Bad Answers: {bad_answers}/{total} = {bad_answers / total:.2%}", file=f)

    
    

# print ("Running Image+text") 
bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]
json_files_list = ["task_3_negated_questions_bird_merged_fixed", "task_3_negated_questions_food_merged_fixed", "task_3_negated_questions_aircraft_merged_fixed", "task_3_negated_questions_dog_merged_fixed", "task_3_negated_questions_car_merged_fixed"]

from datetime import datetime

for json_file, images_folder in zip(json_files_list, folder_lists):

    bird_images = get_bird_images(images_folder)
    json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file}.json"

    json_data = get_json_data(json_file_name)
    # captions_mcqid_pair_dict = get_map_dict(json_data)

    medium_data, hard_data = get_medium_hard_data(json_data)

    
    
    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Medium----", file=f)
    
    run_eval_image_text(medium_data)

    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Hard----", file=f)
    run_eval_image_text(hard_data)

# # Text-Only
def get_answer_text_only(query):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,  # Plain string, no image content
            }
        ],
        model=model,
        max_completion_tokens=32,
    )
    result = response.choices[0].message.content
    match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
    final_answer = match.group(1).strip() if match else result
    return final_answer

def run_text_only_eval(data_partition):

    with open(output_file_name, "a", encoding="utf-8") as f:
        print (f"\n----------Running in Text-only mode----------\n", file=f)


    # Counters for distribution
    true_distribution = Counter()
    predicted_distribution = Counter()
    
    results = []
    bad_answers = 0
    
    for i, item in tqdm(enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        correct_description =  item["correct_description"]
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:5]
    
        # Format the prompt
        formatted_prompt = f"{question}\n"
        # formatted_prompt = f"{question} Ignore the descriptions and focus only on the class names.\n" # 
        # formatted_prompt = f"{question} Ignore the class names and focus on the descriptions.\n" # 
        for k in ['A', 'B', 'C', 'D']:  # ['D', 'C', 'B', 'A'] for position bias checking ['A', 'B', 'C', 'D']
            formatted_prompt += f"{k}. {options[k]}\n"
    
        # Final prompt
        # prompt = f""" First, generate a description of the object in the image. Then, use this for helping to answer the following question. Do not show your reasoning or explanation. Your final answer or response must be a in this format 'Final: A' or 'Final: B' etc. Make sure this is the format of the final answer. 
    
        # {formatted_prompt}
    
        # Answer options are: ('A', 'B', 'C', 'D')"""

        # prompt with correct description caption added.
        prompt = f""" Here is a description of the target object in an image: "{correct_description}"\n. Please answer the following question. Your final answer or response must be in this format 'Final: A' or 'Final: B' etc.
    
        {formatted_prompt}
    
        Do NOT show any reasoning. Answer options are: ('A', 'B', 'C', 'D')"""

        # Do NOT show any reasoning. 
        # print(f"MCQ ID: {mcq_id}")
        # print(f"\n\nPrompt Query :{prompt}")
        
        # Run the model
        model_output = get_answer_text_only(prompt)
        # print(f"Model Output:{model_output}")
        model_output = model_output.replace(" ", "")
        

        # Extract predicted answer (basic string search, can refine)
        predicted_answer = None
        found_at_least_one = 0
        for option in ['A', 'B', 'C', 'D']:
            # if f"{option}" in model_output or f"{option}." in model_output:#orig
            if f"Final:{option}" in model_output or f"Final:{option}." in model_output or f"Final:'{option}'." in model_output or f"'{option}'" in model_output:
                predicted_answer = option
                found_at_least_one = 1
                
            elif len(model_output.strip())<5 and (f"{option}" in model_output or f"{option}." in model_output):
                predicted_answer = option
                found_at_least_one = 1
                
        if found_at_least_one == 0:
            # print (f"model_output: {model_output}")
            # print("Model output not following answer option format")
            predicted_answer = "X"
            bad_answers = bad_answers + 1
            # raise ValueError("Model output not following answer option format")    
        

        

        # Update counters
        true_distribution[correct_answer] += 1
        if predicted_answer:
            predicted_distribution[predicted_answer] += 1
        
        results.append({
            'mcq_id': mcq_id,
            # 'image_path': image_path,
            'prompt': prompt,
            'model_output': model_output,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': predicted_answer == correct_answer
        })

        # if i>10:
        #     break

    
    print(f"Results for file: {json_file_name}")
    # Accuracy summary 
    correct = sum(r['is_correct'] for r in results if r['predicted_answer'] is not None)
    total = len(results)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}") 
    
    # Print distributions
    print("True Option Distribution:", dict(true_distribution))
    print("Predicted Option Distribution:", dict(predicted_distribution))   
    
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Negation: description-only Results for file: {json_file_name}", file=f)
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}", file=f)
        print(f"Bad Answers: {bad_answers}/{total} = {bad_answers / total:.2%}", file=f)

    

bird_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/CUB_200_2011/images"
food_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/food-101/food-101/images"
aircraft_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/fgvc-aircraft-2013b/data/test"
dogs_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-dogs/images/Images"
car_folder = "/projects/abbott_lab/Users/ishtiaque/datasets/stanford-cars/train"

folder_lists = [bird_folder, food_folder, aircraft_folder, dogs_folder, car_folder]
json_files_list = ["task_3_negated_questions_bird_merged_fixed", "task_3_negated_questions_food_merged_fixed", "task_3_negated_questions_aircraft_merged_fixed", "task_3_negated_questions_dog_merged_fixed", "task_3_negated_questions_car_merged_fixed"]




from datetime import datetime

for json_file, images_folder in zip(json_files_list, folder_lists):

    bird_images = get_bird_images(images_folder)
    json_file_name = f"/home/ishtiaqueahmedk/Research/VLM/fine_grained/mcq_json files/{json_file}.json"

    json_data = get_json_data(json_file_name)
    # captions_mcqid_pair_dict = get_map_dict(json_data)

    medium_data, hard_data = get_medium_hard_data(json_data)

    
    
    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Medium----", file=f)
    
    run_text_only_eval(medium_data)

    current_datetime = datetime.now()
    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\ncurrent_datetime: {current_datetime}\n", file=f)
        print("\n----Hard----", file=f)
    run_text_only_eval(hard_data)
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

import csv

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
    
    with open(json_file_name, "r", encoding='utf-8') as file: 
        json_data = json.load(file)
    print(len(json_data))
    return json_data




def get_medium_hard_data(json_data, use_partial = False):
    
    # Use this if the json file contain the medium and hard categories
    medium_data = []
    hard_data = []

    data_counter = 0
    use_partial = False#True
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
        
    # print(len(medium_data))
    # print(len(hard_data))
    return medium_data, hard_data


def run_eval(data_partition, bird_images, get_answer, json_file_name, output_file_name, image_no = 5):

    # Counters for distribution
    distractor_rows = []
    distractor_csv_file_name = output_file_name.replace(".txt", "_similarity.csv")
    
    true_distribution = Counter()
    predicted_distribution = Counter()
    
    results = []
    bad_ans = 0
    
    for i, item in tqdm(enumerate(data_partition)): # for easy part json_data, for medium_data, for hard_data 
        mcq_id = item['mcq_id']
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        similarity_scores = item['similarity_scores']
    
        if mcq_id not in bird_images or not bird_images[mcq_id]:
            print(f"No image for {mcq_id}") 
            continue
    
        image_paths = bird_images[mcq_id][:image_no]

    
        # Run the model
        for image_path in image_paths:

            option_predictions = {}
            for k in ['A', 'B', 'C', 'D']:  

                if (k == correct_answer):
                    correct_answer_bi = 'yes'
                else:
                    correct_answer_bi = 'no'

                correct_answer_bi = correct_answer_bi.lower()
                
        
                # Format the prompt
                if ("task_0b" in json_file_name): #Only ClassName

                    option_text = options[k].replace("(Class:", "").replace(")", "").strip()
                    formatted_prompt = f"Question: Is the object in the image a {option_text}? Answer only 'yes' or 'no'.\n"
                    
                    task_type = "0b"

                elif ("class_descriptions" in json_file_name): #correct Description
                    formatted_prompt = f"Question: Does the object in the image match the visual attributes in this description? '{options[k]}'. Answer only 'yes' or 'no'.\n"
                    
                    task_type = "0c"

                elif ("task_0a" in json_file_name): #correct Description
                    formatted_prompt = f"Question: Does the object in the image match the visual attributes in this description? '{options[k]}'. Answer only 'yes' or 'no'.\n"
                    
                    task_type = "0a"

                else:
                    print ("Invalid File Name!!")
                    raise ValueError 


                # New Yes/NO prompt
                prompt = f""" Your answer or response must ONLY be a single word ('yes', 'no'). Do not respond with any other text and do not show any reasoning. 
                {formatted_prompt}
            
                Answer: ('yes' or 'no')"""


                model_output = get_answer(image_path, prompt)
                model_output = model_output.replace(" ", "")
                model_output = model_output.lower()

                # Extract predicted answer
                found_at_least_one = 0
                predicted_answer = None
                for option in ['yes', 'no', 'unknown']:
                    if f"{option}" in model_output:
                        predicted_answer = option
                        found_at_least_one = 1
                        break
        
                # Update counters
                true_distribution[correct_answer_bi] += 1
                if predicted_answer:
                    predicted_distribution[predicted_answer] += 1

                if found_at_least_one == 0:
                    bad_ans = bad_ans + 1
                    print (f"Model not following output format!! {model_output}")
                
                option_predictions[k] = {
                    'predicted_answer': predicted_answer,
                    'correct_answer_bi': correct_answer_bi,
                    'model_output': model_output,
                    'is_correct': predicted_answer == correct_answer_bi
                }

            
            # Store one CSV row per distractor option
            for k in ['A', 'B', 'C', 'D']:
                if k != correct_answer:
                    distractor_rows.append({
                        'mcq_id': mcq_id,
                        'image_path': image_path,
                        'correct_answer': correct_answer,
                        'distractor_option': k,
                        'similarity_score': similarity_scores[k],
                        'model_prediction': option_predictions[k]['predicted_answer'],
                        'model_output': option_predictions[k]['model_output'],
                        'prediction_correct': int(option_predictions[k]['is_correct']),
                        'task_type': task_type
                    })
            
            all_four_correct = all(
                option_predictions[k]['is_correct']
                for k in ['A', 'B', 'C', 'D']
            )

            first_no_option = next(
                k for k in ['A', 'B', 'C', 'D']
                if k != correct_answer
            )

            two_way_binary_correct = (
                option_predictions[correct_answer]['is_correct'] and
                option_predictions[first_no_option]['is_correct']
            )

            # One-way accuracy: only check the correct answer option,
            # i.e., whether the model correctly answered "yes" for the true option.
            one_way_binary_correct = option_predictions[correct_answer]['is_correct']

            results.append({
                'mcq_id': mcq_id,
                'image_path': image_path,
                'correct_answer': correct_answer,
                'option_predictions': option_predictions,
                'all_four_correct': all_four_correct,
                'first_no_option': first_no_option,
                'two_way_binary_correct': two_way_binary_correct,
                'one_way_binary_correct': one_way_binary_correct
            })
    
    print(f"Results for file: {json_file_name}")

    total = len(results)

    all_four_correct_count = sum(r['all_four_correct'] for r in results)
    two_way_correct_count = sum(r['two_way_binary_correct'] for r in results)
    one_way_correct_count = sum(r['one_way_binary_correct'] for r in results)

    print(f"All-four accuracy: {all_four_correct_count}/{total} = {all_four_correct_count / total:.2%}")
    print(f"Two-way binary accuracy: {two_way_correct_count}/{total} = {two_way_correct_count / total:.2%}")
    print(f"One-way binary accuracy: {one_way_correct_count}/{total} = {one_way_correct_count / total:.2%}")

    # Print distributions
    print("True Option Distribution:", dict(true_distribution))
    print("Predicted Option Distribution:", dict(predicted_distribution)) 

    with open(output_file_name, "a", encoding="utf-8") as f:
        print(f"\n ------- Results for file: {json_file_name}", file=f)
        print(f"All-four accuracy: {all_four_correct_count}/{total} = {all_four_correct_count / total:.2%}", file=f)
        print(f"Two-way binary accuracy: {two_way_correct_count}/{total} = {two_way_correct_count / total:.2%}", file=f)
        print(f"One-way binary accuracy: {one_way_correct_count}/{total} = {one_way_correct_count / total:.2%}", file=f)
        print(f"bad_ans: {bad_ans}/{total * 4} = {bad_ans / (total * 4):.2%}", file=f)
        
    file_exists = os.path.exists(distractor_csv_file_name)
    
    with open(distractor_csv_file_name, "a", encoding="utf-8", newline="") as f:

        fieldnames = [
            'mcq_id',
            'image_path',
            'correct_answer',
            'distractor_option',
            'similarity_score',
            'model_prediction',
            'model_output',
            'prediction_correct',
            'task_type'
        ]
    
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerows(distractor_rows)
        
    print (f"wrote into csv file for file: {json_file_name}")



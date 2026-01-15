import random
import numpy as np
import torch
import math
import argparse
import json
import re
import os

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_jsonl(data: list, file_path: str):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def replace_none_with_mean(batch):
    valid_values = [x for x in batch if x is not None]
    if not valid_values:
        raise ValueError("All scores are None!!! Please check your judge prompt!")
    mean_value = np.mean(valid_values)
    result = [x if x is not None else mean_value for x in batch]
    
    return result

def extract_score(output):
    try:
        lines = output.splitlines()
        for line in lines:
            if line.startswith("Score:"):
                score = line.split(":")[1].strip()
                return float(int(score))    # Float for reward calculation
    except:
        return None

def calculate_standby_reward(output):
    if not output or not isinstance(output, str):
        return None
    pattern = r'total score:\s*(\d+)\s*/\s*\d+'
    lines = output.splitlines()
    
    for line in lines:
        if line.strip():
            match = re.search(pattern, line.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
    return None

def calculate_final_scores(judge_scores, cal_mode: str, weights: dict = None):
    dimensions = ["resonation", "expression", "reception"]
    all_scores = []

    num_samples = len(judge_scores[dimensions[0]])
    
    # Average reward
    if cal_mode == "mean":
        if weights:
            for i in range(num_samples):
                weighted_sum = 0
                total_weight = 0
                
                for dim in dimensions:
                    weighted_sum += judge_scores[dim][i] * weights[dim] / 5.0
                    total_weight += weights[dim]
                
                if total_weight > 0:
                    all_scores.append(weighted_sum / total_weight)
                else:
                    all_scores.append(0.0)
                    
        else:
            for i in range(num_samples):
                total_score = 0
                valid_dims = 0
                
                for dim in dimensions:
                    total_score += judge_scores[dim][i] / 5.0
                    valid_dims += 1
                
                if valid_dims > 0:
                    all_scores.append(total_score / valid_dims)
                else:
                    all_scores.append(0.0)
    
    # Harmonic mean
    elif cal_mode == "harmonic":
        for i in range(num_samples):
            total_score = 0
            valid_dims = 0
            
            for dim in dimensions:
                if judge_scores[dim][i] > 0:
                    total_score += 5.0 / judge_scores[dim][i]
                    valid_dims += 1
            
            if valid_dims > 0:
                all_scores.append(valid_dims / total_score)
            else:
                all_scores.append(0.0)
    
    elif cal_mode == "geometric":
        for i in range(num_samples):
            product = 1.0
            valid_dims = 0
            
            for dim in dimensions:
                normalized_score = judge_scores[dim][i] / 5.0
                if normalized_score > 0:
                    product *= normalized_score
                    valid_dims += 1
            
            if valid_dims > 0:
                geometric_mean = math.pow(product, 1.0 / valid_dims)
                all_scores.append(geometric_mean)
            else:
                all_scores.append(0.0)
    
    
    else:
        raise ValueError(f"We do not support {cal_mode} to calculate final reward.")

    return all_scores

def save_args_to_json(args: argparse.Namespace, output_path: str = None) -> str:
    args_dict = vars(args)
    
    with open(os.path.join(output_path, "args.json"), 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)

def parse_model_output(output_data):
    text = ""
    if isinstance(output_data, str):
        text = output_data
    elif isinstance(output_data, list):
        if len(output_data) > 0 and isinstance(output_data[-1], dict):
            text = output_data[-1].get('content', '')
        else:
            text = str(output_data)
    elif isinstance(output_data, dict):
        text = output_data.get('content', '')
    else:
        text = str(output_data)

    pattern = r'#\s*Analysis\s*(.*?)\s*#\s*Response\s*(.*)'
    
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        analysis_content = match.group(1).strip()
        response_content = match.group(2).strip()
    else:
        analysis_content = ""
        response_content = ""

    return analysis_content, response_content

def extract_features(all_completions):
    all_responses = []
    all_analysis = []
    for completion in all_completions:
        analysis, response = parse_model_output(completion)
        all_analysis.append(analysis)
        all_responses.append(response)
    return all_analysis, all_responses
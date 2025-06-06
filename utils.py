"""
This module provides utility functions for the FinGPT Forecaster project,
including:
- Tokenization of prompts and answers for LLM training.
- Parsing base model names to Hugging Face identifiers.
- Loading datasets from disk or Hugging Face Hub.
- Parsing structured answers from LLM outputs.
- Calculating ROUGE scores and other metrics for evaluation.
"""
import re
import os
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error # For evaluation metrics
from collections import defaultdict
from rouge_score import rouge_scorer # For text generation evaluation

# Dictionary mapping model types to their LoRA target modules.
# This is used for PEFT (Parameter-Efficient Fine-Tuning) configuration.
lora_module_dict = {
    'chatglm2': ['query_key_value'], # Target modules for ChatGLM2
    'llama2': [                     # Target modules for Llama-2 models
        'q_proj', 'k_proj', 'v_proj', # Attention query, key, value projections
        'o_proj',                     # Attention output projection
        'gate_proj', 'up_proj', 'down_proj', # Feed-forward network projections
        # 'embed_tokens', 'lm_head', # Embedding and language model head (often not targeted by LoRA)
    ],
}


def tokenize(args, tokenizer, feature):
    """
    Tokenizes a single data feature (prompt + answer) for supervised fine-tuning.
    Formats input_ids and labels for causal language modeling.

    Args:
        args (Namespace or dict-like): Configuration object containing `max_length`.
        tokenizer: The Hugging Face tokenizer instance.
        feature (dict): A dictionary containing 'prompt' and 'answer' strings.

    Returns:
        dict: A dictionary with 'input_ids', 'labels', and 'exceed_max_length' boolean.
              'labels' are masked for the prompt part (set to pad_token_id).
    """
    # Tokenize the prompt
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), 
        padding=False, # No padding here; will be handled by data collator or later
        max_length=args.max_length, 
        truncation=True
    )
    
    # Tokenize the answer (target)
    # `add_special_tokens=False` because EOS is typically handled at the end of the combined sequence.
    target_ids = tokenizer.encode(
        feature['answer'].strip(), 
        padding=False, 
        max_length=args.max_length, # This max_length applies to the answer alone if prompt is too long
        truncation=True, 
        add_special_tokens=False # Usually, special tokens like EOS are added to the combined sequence
    )
    
    # Combine prompt and target IDs
    input_ids = prompt_ids + target_ids
    
    # Check if the combined length exceeds max_length (after potential truncation of prompt/target individually)
    # This check is a bit redundant if tokenizer handles truncation for the combined sequence,
    # but good for explicitly knowing if combined input was too long.
    # A more accurate check would be len(input_ids) after potential combined truncation.
    # For now, assuming individual truncations handle this enough for this flag.
    # Actually, the input_ids are already formed. So, this check is accurate for the *formed* input_ids.
    exceed_max_length = len(input_ids) >= args.max_length 
    if exceed_max_length: # If combined is too long, truncate the combined sequence
        input_ids = input_ids[:args.max_length]

     # Add End-Of-Sentence (EOS) Token if not already present and sequence is not at max length
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    elif exceed_max_length and input_ids[-1] != tokenizer.eos_token_id : # Ensure EOS if truncated
        input_ids[-1] = tokenizer.eos_token_id


    # Create labels: mask prompt tokens by setting them to pad_token_id
    # The model should only learn to predict the answer part.
    # Length of prompt_ids might change if it was truncated by tokenizer.encode
    # So, it's crucial that prompt_ids here is what makes up the first part of input_ids.
    len_prompt_tokens = len(prompt_ids)
    if len_prompt_tokens > args.max_length: # If prompt itself was truncated
        len_prompt_tokens = args.max_length # It can't be longer than max_length

    # Labels for the prompt part are ignored (masked)
    label_ids = [tokenizer.pad_token_id] * len_prompt_tokens 
    # Labels for the answer part are the token IDs themselves
    label_ids += input_ids[len_prompt_tokens:]
    
    # Ensure labels are not longer than input_ids (can happen if input_ids was truncated)
    label_ids = label_ids[:len(input_ids)]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length # Indicates if original combined prompt+answer was too long
    }


def parse_model_name(name, from_remote=False):
    """
    Maps a short model alias (e.g., 'llama2') to its full Hugging Face model identifier.

    Args:
        name (str): The short alias of the model.
        from_remote (bool): If True, returns the remote Hugging Face Hub identifier.
                            If False, can return a local path (though current impl focuses on remote).

    Returns:
        str: The full model identifier.

    Raises:
        ValueError: If the model alias is not defined.
    """
    if name == 'chatglm2':
        # Example: 'THUDM/chatglm2-6b' or a local path 'base_models/chatglm2-6b'
        return 'THUDM/chatglm2-6b' if from_remote else os.path.join('base_models', 'chatglm2-6b')
    elif name == 'llama2':
        # Llama-2 models are typically loaded from Hugging Face Hub
        return 'meta-llama/Llama-2-7b-chat-hf' # Defaulting to 7b chat HF variant
        # The original code had a commented out local path option:
        # return 'meta-llama/Llama-2-7b-chat-hf' if from_remote else 'base_models/Llama-2-7b-chat-hf'
    else:
        raise ValueError(f"Undefined base model alias: {name}")
        
    
def load_dataset(names, from_remote=False):
    """
    Loads one or more datasets, either from local disk or Hugging Face Hub.
    Handles dataset replication if specified (e.g., "my_dataset*3").
    Ensures datasets have a 'test' split, creating one if missing.

    Args:
        names (str): A comma-separated string of dataset names.
                     Can include replication factor (e.g., "dataset_name*3").
        from_remote (bool): If True, loads from Hugging Face Hub.
                            If False, loads from local disk (or Hub if path not found locally).

    Returns:
        list: A list of Hugging Face `DatasetDict` objects.
    """
    dataset_names_str_list = [d.strip() for d in names.split(',')]
    loaded_dataset_list = []
    
    for name_str in dataset_names_str_list:
        replication_factor = 1
        actual_dataset_name = name_str
        
        # Parse replication factor if present (e.g., "my_data*3")
        if '*' in name_str:
            parts = name_str.split('*')
            actual_dataset_name = parts[0]
            try:
                replication_factor = int(parts[1])
            except ValueError:
                print(f"Warning: Invalid replication factor in '{name_str}'. Defaulting to 1.")
                replication_factor = 1
        
        # Determine full dataset path/identifier
        # If not a local path and `from_remote` is true, or if it's a Hub identifier.
        # The original logic for prepending 'FinGPT/fingpt-forecaster-' seems specific.
        dataset_path_or_id = actual_dataset_name
        if not os.path.exists(actual_dataset_name): # If not a direct local path
            if from_remote: # Assume it's a Hub ID that might need a prefix
                 dataset_path_or_id = f'FinGPT/fingpt-forecaster-{actual_dataset_name}' # Default FinGPT prefix
            # If not from_remote and not a local path, it might be a direct Hub ID.
            # `datasets.load_dataset` can handle Hub IDs directly.
            # The original logic had 'data/fingpt-forecaster-' for local, which implies a specific dir structure.

        # Load the dataset
        try:
            if from_remote or not os.path.exists(dataset_path_or_id): # Try Hub if local path fails or forced remote
                print(f"Loading dataset '{dataset_path_or_id}' from Hugging Face Hub...")
                # If `dataset_path_or_id` was already a valid Hub ID, this works.
                # If it was a prefixed name (like FinGPT/...), it also works.
                current_dataset_dict = datasets.load_dataset(dataset_path_or_id)
            else: # Load from local disk
                print(f"Loading dataset from disk: '{dataset_path_or_id}'...")
                current_dataset_dict = datasets.load_from_disk(dataset_path_or_id)
        except Exception as e:
            print(f"Error loading dataset '{dataset_path_or_id}': {e}. Skipping.")
            continue
            
        # Ensure a 'test' split exists, create one if not (using 20% of train)
        if 'test' not in current_dataset_dict and 'train' in current_dataset_dict:
            print(f"Dataset '{actual_dataset_name}' missing 'test' split. Creating one from 20% of 'train'.")
            # Ensure train split is not empty before splitting
            if len(current_dataset_dict['train']) > 0:
                 # Using a fixed seed for reproducibility if split is created.
                train_test_split = current_dataset_dict['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)
                current_dataset_dict['train'] = train_test_split['train']
                current_dataset_dict['test'] = train_test_split['test']
            else:
                print(f"Warning: Train split for '{actual_dataset_name}' is empty. Cannot create test split.")
                # Create an empty test split with same features if possible
                empty_data_for_test = {feature: [] for feature in current_dataset_dict['train'].features}
                current_dataset_dict['test'] = datasets.Dataset.from_dict(empty_data_for_test, features=current_dataset_dict['train'].features)


        elif 'test' not in current_dataset_dict and 'train' not in current_dataset_dict:
             print(f"Warning: Dataset '{actual_dataset_name}' has no 'train' or 'test' split. Cannot process.")
             continue # Skip this dataset if it's unusable


        loaded_dataset_list.extend([current_dataset_dict] * replication_factor)
    
    return loaded_dataset_list


def parse_answer(answer_text):
    """
    Parses a structured LLM answer string to extract positive developments,
    potential concerns, prediction, and analysis.

    Args:
        answer_text (str): The LLM's answer string.

    Returns:
        dict or None: A dictionary with parsed components ('positive developments', 
                      'potential concerns', 'prediction', 'prediction_binary', 'analysis')
                      if parsing is successful. Returns None otherwise.
    """
    if not isinstance(answer_text, str): # Ensure input is a string
        return None

    # Regex to capture the main sections
    # Using re.IGNORECASE for "Prediction (&|and) Analysis" part
    # Making group capturing non-greedy where appropriate (e.g. (.*?) for developments/concerns)
    main_match = re.match(
        r"^\s*\[Positive Developments\]:\s*(.*?)\s*\[Potential Concerns\]:\s*(.*?)\s*\[Prediction\s*(?:&|and)\s*Analysis\]:\s*(.*)\s*$",
        answer_text,
        flags=re.DOTALL | re.IGNORECASE # DOTALL for multiline, IGNORECASE for "and/&"
    )
    if not main_match:
        # print("Debug: Main structure parse failed.")
        # print(f"Answer snippet: {answer_text[:300]}")
        return None
    
    pros_text = main_match.group(1).strip()
    cons_text = main_match.group(2).strip()
    pna_text = main_match.group(3).strip() # Prediction and Analysis part
        
    # Regex to separate Prediction and Analysis within the PNA part
    # Allow for optional newline between "Prediction:" and the prediction text.
    pna_match = re.match(
        r'^Prediction:\s*(.*?)\s*Analysis:\s*(.*)\s*$', 
        pna_text, 
        flags=re.DOTALL | re.IGNORECASE
    )
    if not pna_match:
        # print("Debug: PNA parse failed.")
        # print(f"PNA text: {pna_text[:300]}")
        return None
        
    prediction_text = pna_match.group(1).strip()
    analysis_text = pna_match.group(2).strip()
        
    # Determine binary prediction (up/down/neutral)
    pred_bin_value = 0 # Default to neutral
    if re.search(r'up|increase|positive', prediction_text, re.IGNORECASE):
        pred_bin_value = 1
    elif re.search(r'down|decrease|decline|negative', prediction_text, re.IGNORECASE):
        pred_bin_value = -1
            
    # Extract numerical prediction margin (e.g., 1-2% -> 1.5, more than 5% -> 5.5)
    # This regex tries to find a percentage range or a single percentage.
    # Example: "up by 1-2%", "down by more than 5%", "increase by 3%"
    pred_margin_value = 0.0
    # Try to match "X-Y%" first
    range_match = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)%', prediction_text)
    if range_match:
        pred_margin_value = (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
    else:
        # Try to match "more than X%" or "X%"
        single_val_match = re.search(r'(?:more\s+than\s+)?(\d+(?:\.\d+)?)\s*%', prediction_text)
        if single_val_match:
            pred_margin_value = float(single_val_match.group(1))
            if "more than" in prediction_text.lower() and pred_margin_value > 0: # Add 0.5 for "more than X%"
                pred_margin_value += 0.5


    # Apply direction to the margin
    pred_margin_final = pred_bin_value * pred_margin_value if pred_bin_value != 0 else 0.0
    # If binary is neutral but margin was found, it's ambiguous, default to 0.
    # If binary is directional but no margin found, margin remains 0.
        
    return {
        "positive developments": pros_text,
        "potential concerns": cons_text,
        "prediction_text_full": prediction_text, # Store original prediction text
        "prediction_numerical": pred_margin_final, # Numerical value (e.g., 1.5, -5.5)
        "prediction_binary": pred_bin_value,     # Binary direction (-1, 0, 1)
        "analysis": analysis_text
    }
    

def calc_rouge_score(references_list, answers_list):
    """
    Calculates average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        references_list (list): A list of reference (ground truth) strings.
        answers_list (list): A list of generated answer strings.

    Returns:
        dict: A dictionary with 'rouge1', 'rouge2', 'rougeL' average F1 scores.
              Returns empty dict if inputs are empty or mismatched.
    """
    if not references_list or not answers_list or len(references_list) != len(answers_list):
        return {}

    scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores for each pair of reference and answer
    scores_per_pair_list = [scorer_instance.score(ref, ans) for ref, ans in zip(references_list, answers_list)]
    
    # Calculate average F1 scores
    # Handle cases where scores_per_pair_list might be empty if input lists were valid but short
    num_pairs = len(scores_per_pair_list)
    if num_pairs == 0:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    avg_rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair_list) / num_pairs
    avg_rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair_list) / num_pairs
    avg_rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair_list) / num_pairs
    
    return {'rouge1': avg_rouge1, 'rouge2': avg_rouge2, 'rougeL': avg_rougeL}

    
def calc_metrics(answers_list_str, ground_truths_list_str):
    
    """
    Calculates and prints evaluation metrics by parsing LLM answers and ground truths.
    Metrics include binary prediction accuracy, MSE for numerical prediction,
    and ROUGE scores for textual parts (developments, concerns, analysis).

    Args:
        answers_list_str (list): List of raw answer strings from the LLM.
        ground_truths_list_str (list): List of raw ground truth answer strings.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    
    # Dictionaries to store parsed components
    parsed_answers_components = defaultdict(list)
    parsed_gts_components = defaultdict(list)
    
    valid_pairs_count = 0
    for raw_answer, raw_gt in zip(answers_list_str, ground_truths_list_str):
        parsed_answer_dict = parse_answer(raw_answer)
        parsed_gt_dict = parse_answer(raw_gt)
        
        # Only consider pairs where both answer and ground truth were successfully parsed
        if parsed_answer_dict and parsed_gt_dict:
            valid_pairs_count += 1
            for key in parsed_answer_dict.keys(): # Iterate through keys like 'positive developments', 'prediction_binary', etc.
                parsed_answers_components[key].append(parsed_answer_dict[key])
                parsed_gts_components[key].append(parsed_gt_dict[key])
    
    if valid_pairs_count == 0 or not parsed_answers_components['prediction_numerical']: # Check if any valid data for metrics
        print("No valid parsed answer/ground-truth pairs found to calculate metrics.")
        return {
            "valid_count": 0, "bin_acc": 0, "mse": float('inf'),
            "pros_rouge_scores": {}, "cons_rouge_scores": {}, "anal_rouge_scores": {}
        }
    
    # Calculate metrics
    binary_accuracy = accuracy_score(parsed_gts_components['prediction_binary'], parsed_answers_components['prediction_binary'])
    mean_squared_err = mean_squared_error(parsed_gts_components['prediction_numerical'], parsed_answers_components['prediction_numerical'])
    
    # Calculate ROUGE scores for textual components
    pros_rouge = calc_rouge_score(parsed_gts_components['positive developments'], parsed_answers_components['positive developments'])
    cons_rouge = calc_rouge_score(parsed_gts_components['potential concerns'], parsed_answers_components['potential concerns'])
    analysis_rouge = calc_rouge_score(parsed_gts_components['analysis'], parsed_answers_components['analysis'])
                              
    # Print the results
    print(f"\nNumber of valid parsed pairs: {valid_pairs_count}")
    print(f"Binary Prediction Accuracy: {binary_accuracy:.4f}")
    print(f"Numerical Prediction Mean Square Error: {mean_squared_err:.4f}")
    print(f"ROUGE Scores for Positive Developments: {pros_rouge}")
    print(f"ROUGE Scores for Potential Concerns: {cons_rouge}")
    print(f"ROUGE Scores for Summary Analysis: {analysis_rouge}")
                              
    return {
        "valid_count": valid_pairs_count,
        "bin_acc": binary_accuracy,
        "mse": mean_squared_err,
        "pros_rouge_scores": pros_rouge,
        "cons_rouge_scores": cons_rouge,
        "anal_rouge_scores": analysis_rouge
    }
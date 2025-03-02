"""
Train a CodeLlama model on software PR data using GRPO for reinforcement learning.
Uses 4-bit quantization to run on consumer hardware like MacBook M2.
"""

import os
import json
import torch
import difflib
from pathlib import Path
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training, 
    get_peft_model,
    PeftConfig,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import GRPO

# Load environment variables
load_dotenv()

# HuggingFace token (for pushing models)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Configuration
class Config:
    # Model settings
    base_model = "codellama/CodeLlama-7b-hf"  # Base model
    model_max_length = 4096  # Context window size
    push_to_hub = True if HF_TOKEN else False
    model_id = "your-username/codellama-swe-rl"  # Change to your username
    
    # LoRA settings
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    
    # Training settings
    batch_size = 2  # Small batch size for MacBook
    micro_batch_size = 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    num_epochs = 1
    learning_rate = 1e-5
    optimizer = "adamw_torch"
    lr_scheduler = "cosine"
    max_steps = 1000
    
    # Data settings
    data_path = "processed_pr_data"
    
    # GRPO settings
    group_size = 4  # Number of candidates per input
    reward_strategy = "average"  # How to combine rewards
    
    # Save/checkpoint settings
    save_steps = 50
    logging_steps = 10
    eval_steps = 100

def format_prompt(example):
    """Format an example into a prompt for the model"""
    prompt = f"""We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{example['problem_statement']}
--- END ISSUE ---

Below are some code segments from a relevant file:
--- BEGIN FILE ---
```
{example['content']}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.
Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
"""
    return prompt

def parse_edits(text):
    """Parse search/replace edits from model output"""
    edits = []
    lines = text.split('\n')
    
    # State tracking
    current_file = None
    in_search = False
    in_replace = False
    current_edit = {"file": None, "search": "", "replace": ""}
    
    for line in lines:
        if "<<<<<<< SEARCH" in line:
            in_search = True
            in_replace = False
            # Try to find the file path in previous lines
            for i in range(len(edits), 0, -1):
                if "###" in lines[i-1]:
                    current_file = lines[i-1].strip("# ")
                    break
            current_edit["file"] = current_file
            continue
        elif "=======" in line:
            in_search = False
            in_replace = True
            continue
        elif ">>>>>>> REPLACE" in line:
            in_search = False
            in_replace = False
            if current_edit["file"] and current_edit["search"] and current_edit["replace"]:
                edits.append(current_edit.copy())
            current_edit = {"file": None, "search": "", "replace": ""}
            continue
        
        if in_search:
            current_edit["search"] += line + "\n"
        elif in_replace:
            current_edit["replace"] += line + "\n"
    
    return edits

def compute_reward(predicted_edits, ground_truth_edits):
    """Compute similarity between predicted and ground truth edits"""
    if not predicted_edits:
        return -1.0  # Penalty for no edits
    
    # Compare each predicted edit with each ground truth edit
    similarities = []
    for gt_edit in ground_truth_edits:
        best_sim = 0
        for pred_edit in predicted_edits:
            # Compare search parts
            search_sim = difflib.SequenceMatcher(
                None, pred_edit["search"], gt_edit["search"]
            ).ratio()
            
            # Compare replace parts
            replace_sim = difflib.SequenceMatcher(
                None, pred_edit["replace"], gt_edit["replace"]
            ).ratio()
            
            # Average the similarities
            sim = (search_sim + replace_sim) / 2
            best_sim = max(best_sim, sim)
        
        if best_sim > 0:
            similarities.append(best_sim)
    
    if not similarities:
        return 0.0
    
    # Return average similarity across all edits
    return sum(similarities) / len(similarities)

def prepare_dataset(data_path):
    """Load and prepare the dataset"""
    if not Path(data_path).exists():
        # If processed data doesn't exist, process raw PR data
        print("Processed data not found, preparing from raw PR data...")
        process_raw_data()
    
    # Load dataset
    try:
        dataset = load_from_disk(data_path)
        print(f"Loaded dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_raw_data():
    """Process raw PR data to create training dataset"""
    from process_data import process_pr_data
    
    # Process PR data
    print("Processing PR data...")
    dataset = process_pr_data()
    
    # Save processed dataset
    dataset.save_to_disk(Config.data_path)
    print(f"Saved processed dataset to {Config.data_path}")

def load_quantized_model():
    """Load a 4-bit quantized model with PEFT/LoRA for efficient training"""
    print(f"Loading model: {Config.base_model}")
    
    # Quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Pad on left for better causal LM
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def reward_function(prompts, responses, rewards=None):
    """Reward function for GRPO"""
    batch_rewards = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # Extract example from prompt (this would need to be adapted based on your data)
        example_id = i % (len(prompts) // Config.group_size)
        example = train_dataset[example_id]
        
        # Parse edits from response
        try:
            predicted_edits = parse_edits(response)
            ground_truth_edits = example['edits']
            
            # Compute reward
            reward = compute_reward(predicted_edits, ground_truth_edits)
            batch_rewards.append(reward)
        except Exception as e:
            print(f"Error computing reward: {e}")
            batch_rewards.append(-1.0)
    
    return batch_rewards

def train():
    """Main training function"""
    global train_dataset  # Make accessible to reward function
    
    # Prepare dataset
    train_dataset = prepare_dataset(Config.data_path)
    if train_dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_quantized_model()
    
    # Set up GRPO
    trainer = GRPO(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config={"lora_alpha": Config.lora_alpha, "lora_dropout": Config.lora_dropout, "r": Config.lora_r},
        max_prompt_length=Config.model_max_length // 2,  # Half for prompt, half for completion
        max_completion_length=Config.model_max_length // 2,
        group_size=Config.group_size,
        logging_dir="./logs",
        project_kwargs={
            "logging_steps": Config.logging_steps,
            "save_strategy": "steps",
            "save_steps": Config.save_steps,
            "evaluation_strategy": "steps",
            "eval_steps": Config.eval_steps,
            "per_device_train_batch_size": Config.micro_batch_size,
            "gradient_accumulation_steps": Config.gradient_accumulation_steps,
            "num_train_epochs": Config.num_epochs,
            "max_steps": Config.max_steps,
            "learning_rate": Config.learning_rate,
            "optim": Config.optimizer,
            "lr_scheduler_type": Config.lr_scheduler,
            "fp16": True,
            "push_to_hub": Config.push_to_hub,
            "hub_model_id": Config.model_id if Config.push_to_hub else None,
        }
    )
    
    # Start training with reward function
    trainer.train(
        formatting_func=format_prompt,
        reward_func=reward_function,
    )
    
    # Save final model
    if not Config.push_to_hub:
        trainer.model.save_pretrained("./codellama-swe-rl")
        tokenizer.save_pretrained("./codellama-swe-rl")
        print("Model saved locally to ./codellama-swe-rl")
    else:
        print(f"Model pushed to HuggingFace Hub: {Config.model_id}")

if __name__ == "__main__":
    train()
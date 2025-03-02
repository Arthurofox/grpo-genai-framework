import torch
import os
from datasets import load_from_disk, Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import GRPOTrainer, GRPOConfig
import swerl  # Local import from the research paper repository
from swerl.core.prompts import AGENTLESS_REPAIR
from swerl.core.reward import calculate_search_replace_reward
# Configuration
class Config:
    """Training configuration and hyperparameters."""
    base_model = "NousResearch/Hermes-3-Llama-3.2-3B"  # Base CodeLlama model
    data_path = "processed_pr_data"  # Path to your dataset
    micro_batch_size = 4  # Small batch size for limited memory
    gradient_accumulation_steps = 4  # Accumulate gradients over steps
    learning_rate = 1e-4  # Learning rate for optimization
    logging_steps = 10  # Log every 10 steps
    num_epochs = 1  # Number of training epochs
    save_steps = 50  # Save model every 50 steps
    group_size = 4  # Number of completions per prompt for GRPO
    optimizer = "adamw_torch"  # Optimizer type
    lr_scheduler = "cosine"  # Learning rate scheduler type
    lora_r = 16  # LoRA rank
    lora_alpha = 32  # LoRA scaling factor
    lora_dropout = 0.05  # Dropout rate for LoRA
    push_to_hub = False  # Whether to push to Hugging Face Hub
    model_id = "your-username/codellama-swe-rl"  # Hub model ID if pushing
    max_steps = 5  # Max steps for quick testing; increase for full training
    model_max_length = 512  # Maximum sequence length

def get_dataset_path(data_path):
    """Get the absolute path to the dataset."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    return project_root / data_path

def prepare_dataset(data_path):
    """
    Load and prepare the dataset with the required fields.
    
    Args:
        data_path (str): Path to the dataset directory.
    
    Returns:
        Dataset: Processed dataset with required fields.
    """
    full_path = get_dataset_path(data_path)
    print(f"Loading dataset from: {full_path}")
    
    # Attempt to load the dataset
    try:
        dataset = load_from_disk(str(full_path))
        print(f"Successfully loaded dataset with {len(dataset)} examples")
        print(f"Dataset columns: {dataset.column_names}")
        
        # Check dataset structure
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def format_prompt(example):
    """
    Format the input prompt for the model using the official SWE-RL prompt template.
    
    Args:
        example (dict): Dataset example with problem statement and content.
    
    Returns:
        str: Formatted prompt string.
    """
    return AGENTLESS_REPAIR.format(
        problem_statement=example['problem_statement'],
        content=example['content']
    )


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
            for i in range(len(edits), len(lines)):
                if i < 0:
                    continue
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
    """
    Compute similarity between predicted and ground truth edits using SWE-RL approach.
    
    Args:
        predicted_edits (list): List of predicted edit dictionaries.
        ground_truth_edits (list): List of ground truth edit dictionaries.
        
    Returns:
        float: Reward value between -1.0 and 1.0
    """
    try:
        # Use the SWE-RL reward function
        reward, _ = calculate_search_replace_reward(predicted_edits, ground_truth_edits)
        return reward
    except Exception as e:
        print(f"Error in swerl reward calculation: {e}")
        # Fall back to our implementation if needed
        # ... (your existing difflib implementation)

def process_dataset_for_grpo(dataset, tokenizer):
    """Process dataset to make it compatible with GRPO training."""
    processed_data = []
    
    for example in dataset:
        # Format prompt for this example using SWE-RL's prompt template
        prompt = AGENTLESS_REPAIR.format(
            problem_statement=example['problem_statement'],
            content=example['content']
        )
        
        # Create processed example with required "prompt" field for GRPOTrainer
        processed_example = {
            "prompt": prompt,
            "edits": example["edits"],
            "filename": example["filename"],
            # Include any other fields you need for the reward function
        }
        
        processed_data.append(processed_example)
    
    return Dataset.from_list(processed_data)

def load_model():
    """
    Load model optimized for Apple M2 with LoRA for efficient fine-tuning.
    
    Returns:
        tuple: (model, tokenizer) - Loaded model and tokenizer.
    """
    print(f"Loading model: {Config.base_model}")
    
    # Load model for Apple Silicon
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        device_map="mps",  # Use Metal Performance Shaders
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Configure LoRA for efficient fine-tuning
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
            #"gate_proj", 
            #"up_proj", 
            #"down_proj"
        ]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

class CustomDataCollator:
    """Custom data collator that handles the dataset format for GRPO training."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        # If the examples are already in the right format, return them
        if isinstance(examples, dict):
            return examples
        
        batch = {
            "input_ids": [example["input_ids"] for example in examples],
            "attention_mask": [example["attention_mask"] for example in examples],
        }
        
        # Add metadata for reward function
        batch["metadata"] = examples
        
        return batch

def train():
    """Main function to execute the SWE-RL training process."""
    # Load the dataset
    dataset = prepare_dataset(Config.data_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Get a small sample for testing on M2
    if Config.max_steps <= 10:
        print(f"Using a small sample of {min(20, len(dataset))} examples for testing")
        dataset = dataset.select(range(min(20, len(dataset))))
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Process dataset for GRPO
    processed_dataset = process_dataset_for_grpo(dataset, tokenizer)
    
    # Define reward function for GRPO
    def reward_function(completions, **kwargs):
        print(f"Reward function called with {len(completions)} completions")
        rewards = []
        for i, completion in enumerate(completions):
            try:
                # Get the original example
                batch_idx = i // Config.group_size
                example = kwargs.get("metadata", [None])[batch_idx % len(kwargs.get("metadata", []))]
                
                # Parse the generated edits
                predicted_edits = parse_edits(completion)
                
                # Get ground truth edits
                ground_truth_edits = example["edits"] if example else []
                
                # Compute reward
                reward = compute_reward(predicted_edits, ground_truth_edits)
                rewards.append(reward)
            except Exception as e:
                print(f"Error in reward calculation: {e}")
                rewards.append(-1.0)
        
        return rewards
    
    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./codellama-swe-rl",
        per_device_train_batch_size=Config.micro_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        logging_steps=Config.logging_steps,
        num_train_epochs=Config.num_epochs,
        dataloader_num_workers=0,  # No multi-processing for debugging
        max_steps=Config.max_steps,
        save_steps=Config.save_steps,
        num_generations=Config.group_size,
        optim=Config.optimizer,
        lr_scheduler_type=Config.lr_scheduler,
        fp16=False,  # Better compatibility with MPS
        push_to_hub=Config.push_to_hub,
        hub_model_id=Config.model_id if Config.push_to_hub else None,
        remove_unused_columns=False,
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    if not Config.push_to_hub:
        model_path = "./codellama-swe-rl"
        trainer.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved locally to {model_path}")
    else:
        print(f"Model pushed to Hugging Face Hub: {Config.model_id}")

if __name__ == "__main__":
    train()
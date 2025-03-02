"""
Train a CodeLlama model on software PR data using GRPO for reinforcement learning.
Uses fp16 on MacBook M2.
"""

import os
from pathlib import Path
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import torch

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

class Config:
    base_model = "codellama/CodeLlama-7b-hf"
    model_max_length = 4096
    push_to_hub = True if HF_TOKEN else False
    model_id = "Kroalist/codellama-swe-rl-test1"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    batch_size = 4
    micro_batch_size = 4
    gradient_accumulation_steps = 1
    num_epochs = 1
    learning_rate = 1e-5
    optimizer = "adamw_torch"
    lr_scheduler = "cosine"
    max_steps = 1000
    data_path = "processed_pr_data"
    group_size = 4
    reward_strategy = "average"
    save_steps = 50
    logging_steps = 1
    eval_steps = 100

def format_prompt(example):
    return f"""We are currently solving the following issue within our repository. Here is the issue text:
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


class CustomProcessing:
    def __init__(self, tokenizer, format_func, max_length=512):
        self.tokenizer = tokenizer
        self.format_func = format_func
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _process_single(self, ex, **kwargs):
        if isinstance(ex, dict):
            prompt = self.format_func(ex)
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",  # Pad to max_length for uniform length
                return_tensors="pt",   # Return PyTorch tensors
            )
            ex["prompt"] = prompt
            # Squeeze to remove batch dimension (from [1, seq_len] to [seq_len])
            ex.update({k: v.squeeze(0) for k, v in tokenized.items()})
            return ex
        elif isinstance(ex, str):
            tokenized = self.tokenizer(
                ex,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return tokenized
        else:
            raise ValueError(f"Unsupported input type: {type(ex)}")

    def __call__(self, example, **kwargs):
        if isinstance(example, list):
            if all(isinstance(ex, str) for ex in example):
                return self.tokenizer(
                    example,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                return [self._process_single(ex, **kwargs) for ex in example]
        else:
            return self._process_single(example, **kwargs)
from transformers import DataCollatorWithPadding
import torch

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Extract only relevant columns
        model_inputs = [
            {k: f[k] for k in ["input_ids", "attention_mask"] if k in f}
            for f in features
        ]
        # Pad and stack the inputs
        batch = super().__call__(model_inputs)
        # Optionally, add other columns back as lists if needed by the trainer
        # (GRPOTrainer passes the full batch to reward_function, but here we focus on model inputs)
        return batch

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
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    # Full path to processed data directory
    full_data_path = project_root / data_path
    
    if not full_data_path.exists():
        # If processed data doesn't exist, process raw PR data
        print(f"Processed data not found at {full_data_path}, preparing from raw PR data...")
        process_raw_data()
    
    # Load dataset
    try:
        print(f"Attempting to load dataset from {full_data_path}")
        dataset = load_from_disk(str(full_data_path))
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
    """Load a model optimized for Apple Silicon M2"""
    print(f"Loading model: {Config.base_model}")
    
    # Load base model with default settings for Apple Silicon
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        device_map={"": "mps"},
        torch_dtype=torch.float16,  # Use fp16 instead of 4-bit quantization
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Use LoRA for parameter-efficient fine-tuning
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

from transformers import DataCollatorWithPadding  # Ensure this import is present

def train():
    global train_dataset  # Keep this so the reward function can access the original data

    # Load and prepare the dataset
    train_dataset = prepare_dataset(Config.data_path)
    if train_dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    # Load the quantized model and tokenizer
    model, tokenizer = load_quantized_model()
    #custom_processor = CustomProcessing(tokenizer, format_prompt, max_length=Config.model_max_length)

    # Process the dataset once and keep all columns
    print("Processing dataset...")
    custom_processor = CustomProcessing(tokenizer, format_prompt, max_length=512)
    mapped_examples = [custom_processor(ex) for ex in train_dataset]
    train_dataset = Dataset.from_list(mapped_examples)
    train_dataset = train_dataset.with_format("python")

    # Set format to python (since reward function needs it)
    train_dataset = train_dataset.with_format("python")

    # Debug: Check what we’ve got
    print("Dataset columns:", train_dataset.column_names)
    sample = train_dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Sample input_ids type:", type(sample["input_ids"]))

    # Define the reward function
    def reward_function(completions, **kwargs):
        rewards = []
        for idx, completion in enumerate(completions):
            try:
                predicted_edits = parse_edits(completion)
                example_idx = kwargs.get("example_idx", idx)
                example = train_dataset[example_idx % len(train_dataset)]  # Use original dataset
                ground_truth_edits = example["edits"]
                reward = compute_reward(predicted_edits, ground_truth_edits)
                rewards.append(reward)
            except Exception as e:
                print(f"Error computing reward: {e}")
                rewards.append(-1.0)
        return rewards

    # Check if we can use fp16 (MPS doesn’t always play nice)
    use_fp16 = torch.backends.mps.is_available() and False  # Force False for stability

    # GRPO config—disable column removal since filtering fucked us
    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./codellama-swe-rl",
        per_device_train_batch_size=Config.micro_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        logging_steps=Config.logging_steps,
        num_train_epochs=Config.num_epochs,
        dataloader_num_workers=4,
        max_steps=5,
        save_steps=Config.save_steps,
        num_generations=Config.group_size,
        optim=Config.optimizer,
        lr_scheduler_type=Config.lr_scheduler,
        fp16=False,  # Disabled for MPS
        push_to_hub=Config.push_to_hub,
        hub_model_id=Config.model_id if Config.push_to_hub else None,
        remove_unused_columns=False,  # Keep all columns
    )

    # Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=custom_processor,
        reward_funcs=reward_function,
    )

    # Set custom data collator
    trainer.data_collator = CustomDataCollator(tokenizer=tokenizer)

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the model
    if not Config.push_to_hub:
        trainer.model.save_pretrained("./codellama-swe-rl")
        tokenizer.save_pretrained("./codellama-swe-rl")
        print("Model saved locally to ./codellama-swe-rl")
    else:
        print(f"Model pushed to HuggingFace Hub: {Config.model_id}")

if __name__ == "__main__":
    train()
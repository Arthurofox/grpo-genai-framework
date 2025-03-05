import torch
import os
import json
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb
import numpy as np
from functools import partial
from huggingface_hub import login  # for pushing to HF Hub

# Import the already defined reward and prompt functions
from swerl.core.prompts import THINKING_SYSTEM, AGENTLESS_REPAIR
from swerl.core.reward import calculate_search_replace_reward, apply_code_change

torch.cuda.empty_cache()

# --- Configuration ---
class SWE_RL_Config:
    # Model configuration
    base_model = "NousResearch/Hermes-3-Llama-3.2-3B"  # Base model
    model_max_length = 4096                            # Maximum sequence length
    
    # Dataset configuration
    data_path = "/content/drive/MyDrive/seed_rl_dataset"  # Path to preprocessed dataset
    data_limit = 300    # Set to None for full dataset or an integer for debugging
    
    # LoRA configuration
    use_lora = True         # Whether to use LoRA
    lora_r = 16             # LoRA rank
    lora_alpha = 32         # LoRA scaling parameter
    lora_dropout = 0.05     # LoRA dropout
    
    # Training hyperparameters
    learning_rate = 1e-5    # Learning rate for training
    weight_decay = 0.01     # Weight decay for optimizer
    beta = 0.04           # KL penalty coefficient (β in the paper)
    epsilon = 0.2         # Clipping parameter (ε in the paper)
    group_size = 8        # Number of generations per prompt (G in the paper)
    warmup_steps = 100    # Learning rate warmup steps
    max_steps = 1000      # Maximum training steps
    gradient_accumulation_steps = 4  # Gradient accumulation steps
    per_device_batch_size = 1        # Batch size per device
    max_prompt_length = 2048         # Maximum prompt length
    max_completion_length = 1024     # Maximum completion length
    
    # Generation settings
    temperature = 0.9     # Sampling temperature
    top_p = 0.95          # Nucleus sampling parameter
    
    # Optimization settings
    bf16 = True                         # Use bfloat16 precision
    gradient_checkpointing = True       # Use gradient checkpointing
    
    # Evaluation and logging
    logging_steps = 10      # Log every N steps
    eval_steps = 100        # Evaluate every N steps
    save_steps = 200        # Save model every N steps
    
    # Output directory
    output_dir = "/content/swe-rl-model"  # Directory to save model

    # Hugging Face Hub settings
    hf_repo_id = "Kroalist/codellama-swe-rl-test1"  # Repository name on HF Hub
    hf_api_key = "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"             # Place your Hugging Face API key here


# --- Dataset Loading and Processing ---
def load_seed_dataset(config):
    """
    Load the SWE-RL seed dataset that was processed using process_pr_data_for_swerl.
    """
    print(f"Loading dataset from: {config.data_path}")
    
    # Try JSON first
    json_path = Path(config.data_path) / "seed_rl_dataset.json"
    if json_path.exists():
        print(f"Loading JSON dataset from {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from JSON dataset")
        return dataset
    
    # Otherwise, try HuggingFace dataset
    hf_path = Path(config.data_path) / "hf_dataset"
    if hf_path.exists():
        print(f"Loading HuggingFace dataset from {hf_path}")
        dataset = load_from_disk(str(hf_path))
        dataset = dataset["train"].to_list()
        print(f"Loaded {len(dataset)} examples from HuggingFace dataset")
        return dataset
    
    raise FileNotFoundError(f"Could not find dataset at {config.data_path}")

def form_prompt(issue, ctx):
    """
    Format prompt using the imported prompt templates.
    """
    context_str = ""
    for filename, content in ctx.items():
        context_str += f"--- BEGIN FILE ---\n```\n{content}\n```\n--- END FILE ---\n\n"
    
    user_prompt = AGENTLESS_REPAIR.format(
        problem_statement=issue,
        content=context_str
    )
    
    return {
        "system": THINKING_SYSTEM,
        "user": user_prompt
    }

# --- Model Setup ---
def setup_model_and_tokenizer(config):
    """
    Setup the model and tokenizer for training.
    """
    print(f"Setting up model: {config.base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if config.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if config.use_lora:
        print("Applying LoRA to the model")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

# --- GRPO Training Implementation ---
class GRPOTrainer:
    def __init__(self, model, tokenizer, config):
        """
        Initialize GRPO trainer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Create reference model
        self.ref_model = self._create_ref_model()
        self.optimizer = self._create_optimizer()
        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
        wandb.init(project="swe-rl", config=vars(config))
    
    def _create_ref_model(self):
        print("Creating reference model")
        if hasattr(self.model, "base_model"):
            from copy import deepcopy
            ref_model = deepcopy(self.model)
        else:
            ref_model = type(self.model).from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
                device_map="auto"
            )
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
    
    def _create_optimizer(self):
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and "bias" not in n],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and "bias" in n],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(param_groups, lr=self.config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    def compute_logprobs(self, model, input_ids, attention_mask):
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits[:, :-1, :]  # remove last token
            target_ids = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            gathered_logprobs = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
            return gathered_logprobs
    
    def generate_completions(self, prompts):
        self.model.eval()
        all_completions = []
        batch_size = self.config.per_device_batch_size
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            formatted_texts = []
            for prompt in batch_prompts:
                formatted_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_texts.append(formatted_text)
            tokenized_prompts = self.tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_length
            ).to(self.model.device)
            prompt_length = tokenized_prompts.input_ids.shape[1]
            with torch.no_grad():
                for _ in range(batch_size):
                    batch_completions = []
                    for _ in range(self.config.group_size):
                        outputs = self.model.generate(
                            **tokenized_prompts,
                            max_new_tokens=self.config.max_completion_length,
                            do_sample=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                            num_return_sequences=1
                        )
                        for output in outputs:
                            completion_ids = output[prompt_length:]
                            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                            batch_completions.append(completion)
                    all_completions.append(batch_completions)
        return all_completions
    
    def compute_rewards(self, completions, data_items):
        rewards = []
        for i, item_completions in enumerate(completions):
            data_item = data_items[i]
            ctx = data_item["ctx"]
            patchgt = data_item["patchgt"]
            item_rewards = []
            for completion in item_completions:
                reward, _ = calculate_search_replace_reward(ctx, patchgt, completion)
                item_rewards.append(float(reward))
            rewards.append(item_rewards)
        return rewards
    
    def grpo_step(self, data_batch):
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.model.device)  # ensure tensor type
        metrics = {"rewards": [], "kl": [], "advantages": []}
        
        # Format prompts for each data item
        prompts = [form_prompt(item["issue"], item["ctx"]) for item in data_batch]
        completions = self.generate_completions(prompts)
        rewards = self.compute_rewards(completions, data_batch)
        metrics["rewards"] = [r for sublist in rewards for r in sublist]
        
        for i, (prompt, item_completions, item_rewards) in enumerate(zip(prompts, completions, rewards)):
            if len(set(item_rewards)) <= 1:
                continue  # skip if no learning signal
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            tokenized_prompt = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            prompt_length = tokenized_prompt.input_ids.shape[1]
            
            rewards_tensor = torch.tensor(item_rewards, device=self.model.device)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_reward) / std_reward
            metrics["advantages"].extend(advantages.tolist())
            
            for j, (completion, advantage) in enumerate(zip(item_completions, advantages)):
                full_text = formatted_prompt + completion
                tokenized = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_prompt_length + self.config.max_completion_length
                ).to(self.model.device)
                
                policy_logprobs = self.compute_logprobs(self.model, tokenized.input_ids, tokenized.attention_mask)
                with torch.no_grad():
                    ref_logprobs = self.compute_logprobs(self.ref_model, tokenized.input_ids, tokenized.attention_mask)
                
                completion_mask = torch.zeros_like(policy_logprobs, dtype=torch.bool)
                completion_mask[:, prompt_length-1:] = True
                
                kl_div = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1
                masked_kl = (kl_div * completion_mask).sum() / torch.clamp(completion_mask.sum(), min=1)
                metrics["kl"].append(masked_kl.item())
                
                ratio = torch.exp(policy_logprobs - ref_logprobs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2)
                kl_penalty = self.config.beta * kl_div
                full_loss = (policy_loss + kl_penalty) * completion_mask
                per_token_loss = full_loss.sum() / torch.clamp(completion_mask.sum(), min=1)
                
                total_loss = total_loss + per_token_loss / (self.config.group_size * len(data_batch))
        
        # If no loss was accumulated, force gradient flow with a dummy term.
        if not total_loss.requires_grad:
            total_loss = total_loss + 0 * sum(p.sum() for p in self.model.parameters())
        
        return total_loss, metrics
    
    def train(self, dataset, num_epochs=1):
        os.makedirs(self.config.output_dir, exist_ok=True)
        global_step = 0
        best_reward = -float('inf')
        
        # Main training loop
        for epoch in range(num_epochs):
            indices = torch.randperm(len(dataset)).tolist()
            shuffled_dataset = [dataset[i] for i in indices]
            for i in range(0, len(shuffled_dataset), self.config.per_device_batch_size):
                if global_step >= self.config.max_steps:
                    break
                batch = shuffled_dataset[i:i+self.config.per_device_batch_size]
                loss, metrics = self.grpo_step(batch)
                loss.backward()
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if global_step % self.config.logging_steps == 0:
                    mean_reward = np.mean(metrics["rewards"]) if metrics["rewards"] else -1.0
                    mean_kl = np.mean(metrics["kl"]) if metrics["kl"] else 0.0
                    metrics_dict = {
                        "loss": loss.item(),
                        "mean_reward": mean_reward,
                        "mean_kl": mean_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": global_step,
                    }
                    print(f"Step {global_step}: loss={loss.item():.4f}, reward={mean_reward:.4f}")
                    wandb.log(metrics_dict)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        self.save_model(f"{self.config.output_dir}/best_model")
                
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.config.output_dir}/checkpoint-{global_step}")
                
                global_step += 1
        
        self.save_model(self.config.output_dir)
        wandb.finish()
    
    def save_model(self, output_dir):
        print(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # --- Push to Hugging Face Hub ---
        try:
            self.model.push_to_hub(self.config.hf_repo_id, use_temp_dir=True)
            self.tokenizer.push_to_hub(self.config.hf_repo_id, use_temp_dir=True)
            print(f"Model successfully pushed to HF Hub repo: {self.config.hf_repo_id}")
        except Exception as e:
            print(f"Failed to push model to HF Hub: {e}")

# --- Main Training Function ---
def train_swe_rl():
    config = SWE_RL_Config()
    
    # Log in to Hugging Face using your API key (if provided)
    if config.hf_api_key and config.hf_api_key != "your_hf_api_key_here":
        login(token=config.hf_api_key)
        print("Logged in to Hugging Face Hub")
    else:
        print("HF API key not provided. Model will not be pushed to the Hub.")
    
    dataset = load_seed_dataset(config)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Limit the data for debugging if data_limit is set
    if config.data_limit is not None:
        dataset = dataset[:config.data_limit]
        print(f"Using {len(dataset)} examples for training (data_limit set)")
    
    model, tokenizer = setup_model_and_tokenizer(config)
    trainer = GRPOTrainer(model, tokenizer, config)
    trainer.train(dataset, num_epochs=1)
    print("Training complete!")
    print(f"Model saved to {config.output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_swe_rl()

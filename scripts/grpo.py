import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Any
import difflib
import numpy as np
from tqdm.auto import tqdm

class GRPOTrainer:
    def __init__(
        self,
        model_name: str,
        beta: float = 0.04,
        epsilon: float = 0.2,
        learning_rate: float = 1e-6,
        group_size: int = 8,
        max_prompt_length: int = 1024,
        max_completion_length: int = 512,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GRPO trainer for SWE-RL.
        
        Args:
            model_name: Name or path of the model to fine-tune
            beta: KL coefficient for GRPO
            epsilon: Clipping parameter for GRPO
            learning_rate: Learning rate for the optimizer
            group_size: Number of outputs to generate per prompt (G in the paper)
            max_prompt_length: Maximum length of the prompt
            max_completion_length: Maximum length of generated completions
            gradient_accumulation_steps: Number of steps to accumulate gradients
            device: Device to use for training (cuda or cpu)
        """
        self.beta = beta
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.group_size = group_size
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create reference model (clone of policy model)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move models to device
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Freeze reference model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Create optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Import reward function 
        from swerl.core.reward import calculate_search_replace_reward
        self.calculate_reward = calculate_search_replace_reward
        
        # Import prompt formatter
        from swerl.core.prompts import THINKING_SYSTEM, AGENTLESS_REPAIR
        self.system_prompt = THINKING_SYSTEM
        self.user_prompt_template = AGENTLESS_REPAIR

    def form_prompt(self, issue: str, ctx: Dict[str, str]) -> str:
        """
        Format the prompt following the template in Figure 3 of the paper.
        
        Args:
            issue: Issue description
            ctx: Dictionary of file paths to file contents
            
        Returns:
            Formatted prompt string
        """
        # Format context (files) for the prompt
        context_str = ""
        for filename, content in ctx.items():
            context_str += f"--- BEGIN FILE ---\n```\n{content}\n```\n--- END FILE ---\n\n"
            
        # Apply the prompt template
        prompt = self.user_prompt_template.format(
            problem_statement=issue,
            content=context_str
        )
        
        # Create the full prompt with system and user parts
        full_prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        return full_prompt
    
    def generate_outputs(self, prompt: str, num_outputs: int) -> List[str]:
        """
        Generate multiple outputs for a given prompt.
        
        Args:
            prompt: Formatted prompt string
            num_outputs: Number of outputs to generate (group size G)
            
        Returns:
            List of generated outputs (completions)
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]
        
        # Generate completions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=0.9,
                num_return_sequences=num_outputs,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract generated completions (exclude prompt tokens)
        completions = outputs[:, prompt_length:]
        
        # Decode completions to text
        completion_texts = []
        for completion in completions:
            decoded = self.tokenizer.decode(completion, skip_special_tokens=True)
            completion_texts.append(decoded)
            
        return completion_texts
    
    def compute_logprobs(self, model, input_ids, attention_mask=None):
        """
        Compute log probabilities for each token in the sequence.
        
        Args:
            model: The model to compute logprobs with
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            
        Returns:
            Tensor of log probabilities for each token
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Remove last logit
        
        # Get log probabilities of the actual next tokens
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, 1:]  # Shift right to get targets
        
        # Gather log probs of the actual next tokens
        gathered_logprobs = torch.gather(
            log_probs, 
            dim=2, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return gathered_logprobs
    
    def grpo_step(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform one GRPO optimization step.
        
        Args:
            batch_data: List of data items with issue, ctx, and patchgt
            
        Returns:
            Loss value and metrics dictionary
        """
        self.model.train()
        total_loss = 0
        metrics = {"rewards": [], "advantages": [], "kl": []}
        
        for data_item in batch_data:
            issue = data_item["issue"]
            ctx = data_item["ctx"]
            patchgt = data_item["patchgt"]
            
            # Form prompt q = form-prompt(issue, ctx)
            prompt = self.form_prompt(issue, ctx)
            
            # Generate outputs {o_i}^G_i=1 ~ π_θ_old(· | q)
            outputs = self.generate_outputs(prompt, self.group_size)
            
            # Calculate rewards R(τ) using the reward function
            rewards = []
            for output in outputs:
                reward, _ = self.calculate_reward(ctx, patchgt, output)
                rewards.append(float(reward))
            
            rewards_tensor = torch.tensor(rewards, device=self.device)
            metrics["rewards"].extend(rewards)
            
            # Skip if all rewards are the same (no learning signal)
            if torch.allclose(rewards_tensor, rewards_tensor[0] * torch.ones_like(rewards_tensor)):
                continue
            
            # Tokenize prompt and completions
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = tokenized_prompt["input_ids"].shape[1]
            
            all_inputs = []
            for output in outputs:
                # Combine prompt and completion
                full_text = prompt + output
                tokenized = self.tokenizer(
                    full_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_length + self.max_completion_length
                ).to(self.device)
                all_inputs.append(tokenized)
            
            # Process each output in the group
            batch_loss = 0
            for i, (inputs, reward) in enumerate(zip(all_inputs, rewards_tensor)):
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # Compute log probabilities from both policy and reference models
                with torch.set_grad_enabled(True):
                    policy_logprobs = self.compute_logprobs(self.model, input_ids, attention_mask)
                
                with torch.no_grad():
                    ref_logprobs = self.compute_logprobs(self.ref_model, input_ids, attention_mask)
                
                # Create completion mask (only consider tokens after the prompt)
                completion_mask = torch.zeros_like(policy_logprobs)
                completion_mask[:, prompt_length-1:] = 1
                
                # Compute advantages A_i using normalized rewards within the group
                mean_reward = rewards_tensor.mean()
                std_reward = rewards_tensor.std() + 1e-8
                advantage = (reward - mean_reward) / std_reward
                metrics["advantages"].append(advantage.item())
                
                # Compute KL divergence term D_KL(π_θ || π_ref)
                kl_div = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1
                kl_div = (kl_div * completion_mask).sum() / completion_mask.sum()
                metrics["kl"].append(kl_div.item())
                
                # Calculate the GRPO objective for this output
                # min(π_θ(o_i|q)/π_θ_old(o_i|q) * A_i, clip(π_θ(o_i|q)/π_θ_old(o_i|q), 1-ε, 1+ε) * A_i)
                ratio = torch.exp(policy_logprobs - ref_logprobs.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                
                # Take the minimum of surr1 and surr2
                surr_loss = -torch.min(surr1, surr2)
                
                # Add KL penalty: -β * D_KL(π_θ || π_ref)
                kl_penalty = self.beta * kl_div
                
                # Compute full loss for this output
                output_loss = (surr_loss + kl_penalty) * completion_mask
                output_loss = output_loss.sum() / completion_mask.sum()
                
                # Accumulate loss
                batch_loss += output_loss / self.group_size
            
            # Accumulate batch loss
            total_loss += batch_loss / len(batch_data)
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss, metrics
    
    def train(
        self, 
        train_dataset, 
        num_epochs: int = 3,
        batch_size: int = 4,
        save_steps: int = 100,
        eval_steps: int = 500,
        save_path: str = "./swerl_model"
    ):
        """
        Train the model using GRPO.
        
        Args:
            train_dataset: Dataset containing issue, ctx, and patchgt
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            save_steps: Save model every N steps
            eval_steps: Evaluate model every N steps
            save_path: Path to save the model
        """
        # Create dataloader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                loss, metrics = self.grpo_step(batch)
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_reward": f"{np.mean(metrics['rewards']):.4f}"
                })
                
                global_step += 1
                
                # Save model checkpoint
                if global_step % save_steps == 0:
                    self.model.save_pretrained(f"{save_path}/checkpoint-{global_step}")
                    self.tokenizer.save_pretrained(f"{save_path}/checkpoint-{global_step}")
                
                # Evaluate model
                if global_step % eval_steps == 0:
                    self.evaluate()
        
        # Save final model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
    def evaluate(self, eval_dataset=None, num_examples=10):
        """
        Evaluate the model on a small subset of examples.
        
        Args:
            eval_dataset: Evaluation dataset (optional)
            num_examples: Number of examples to evaluate
        """
        if eval_dataset is None:
            return
        
        print("\nEvaluating model...")
        correct = 0
        
        for i, data_item in enumerate(eval_dataset[:num_examples]):
            issue = data_item["issue"]
            ctx = data_item["ctx"]
            patchgt = data_item["patchgt"]
            
            # Form prompt
            prompt = self.form_prompt(issue, ctx)
            
            # Generate a single output
            output = self.generate_outputs(prompt, 1)[0]
            
            # Calculate reward
            reward, _ = self.calculate_reward(ctx, patchgt, output)
            
            # Count as correct if reward > 0
            if reward > 0:
                correct += 1
                
            print(f"Example {i+1}: Reward = {reward:.4f}")
            
        accuracy = correct / num_examples
        print(f"Evaluation accuracy: {accuracy:.4f} ({correct}/{num_examples})")
        
        return accuracy
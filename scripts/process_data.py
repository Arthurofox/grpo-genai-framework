"""
Process raw PR data into a format suitable for SWE-RL training.
Following the methodology in the SWE-RL paper:
1. Create a seed dataset with issue, code context, and oracle patches
2. Structure data to support GRPO training
3. Ensure patch format consistency for reward calculation
"""

import json
import os
import re
from pathlib import Path
import difflib
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from collections import defaultdict

def process_pr_data_for_swerl(pr_data_dir='pr_data_raw', output_dir='seed_rl_dataset'):
    """
    Process PR data to create a dataset for SWE-RL training.
    Creates the exact data structure needed for the reward function:
    - issue: The issue description
    - ctx: Complete code context (all relevant files)
    - patchgt: The oracle patch for comparison
    """
    print("Processing PR data for SWE-RL training...")
    seed_rl_dataset = []
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    # Full path to PR data directory
    pr_data_path = project_root / pr_data_dir
    output_path = project_root / output_dir
    
    if not pr_data_path.exists():
        raise FileNotFoundError(f"PR data directory not found: {pr_data_path}")
    
    output_path.mkdir(exist_ok=True)
    
    print(f"Reading PR data from {pr_data_path}")
    
    # Group PRs by repository for GRPO sampling
    repos_data = defaultdict(list)
    
    # Count statistics
    total_prs = 0
    valid_prs = 0
    
    # Process all JSON files in the data directory
    for filename in os.listdir(pr_data_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = pr_data_path / filename
        print(f"Processing {file_path}")
        
        with open(file_path, 'r') as f:
            try:
                prs = json.load(f)
                total_prs += len(prs)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")
                continue
        
        # Process each PR
        for pr in tqdm(prs, desc=f"Processing {filename}"):
            # Build the issue description
            issue_desc = ""
            if pr.get('issue_title'):
                issue_desc += f"Issue: {pr['issue_title']}\n\n"
            if pr.get('issue_body'):
                issue_desc += f"{pr['issue_body']}\n\n"
            
            # Skip if no issue description (paper requires linked issues)
            if not issue_desc.strip():
                continue
                
            # Collect all file contents for complete code context
            code_context = {}
            oracle_patches = {}
            
            # Process all files in the PR
            for file_data in pr.get('files', []):
                try:
                    filename = file_data['filename']
                    
                    # Only process programming files
                    if not filename.endswith(('.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.go')):
                        continue
                    
                    # Add file content to context
                    if file_data.get('is_modified', True):
                        code_context[filename] = file_data.get('content_before', '')
                    else:
                        # For unmodified files, use their content directly
                        code_context[filename] = file_data.get('content', '')
                    
                    # Add patch to oracle patches if it exists
                    if file_data.get('patch'):
                        oracle_patches[filename] = file_data['patch']
                    
                except KeyError as e:
                    continue
                except Exception as e:
                    continue
            
            # Skip if no files or patches
            if not code_context or not oracle_patches:
                continue
                
            # Get repo name for grouping
            repo_name = pr.get('repo', os.path.splitext(filename)[0])
            repo_short = repo_name.split('/')[-1] if '/' in repo_name else repo_name
            
            # Create the seed item in the SWE-RL format
            seed_item = {
                'issue': issue_desc,
                'ctx': code_context,
                'patchgt': oracle_patches,
                # Metadata for GRPO
                'repo': repo_short,
                'pr_number': pr.get('number', 0)
            }
            
            # Verify the item has all required components
            if validate_seed_item(seed_item):
                repos_data[repo_short].append(seed_item)
                valid_prs += 1
    
    # Write data grouped by repository (for GRPO sampling)
    for repo, items in repos_data.items():
        print(f"Repository {repo}: {len(items)} examples")
        seed_rl_dataset.extend(items)
        
        # Save each repository's data separately for GRPO sampling
        repo_file = output_path / f"{repo}_seed.json"
        with open(repo_file, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Saved {len(items)} items to {repo_file}")
    
    # Save the complete dataset
    complete_file = output_path / "seed_rl_dataset.json"
    with open(complete_file, 'w') as f:
        json.dump(seed_rl_dataset, f, indent=2)
    print(f"Saved {len(seed_rl_dataset)} total items to {complete_file}")
    
    # Also create a HuggingFace dataset
    create_huggingface_dataset(seed_rl_dataset, output_path)
    
    # Print statistics
    print("\nProcessing complete:")
    print(f"Total PRs: {total_prs}")
    print(f"Valid PRs for SWE-RL training: {valid_prs}")
    print(f"Final dataset size: {len(seed_rl_dataset)} examples")
    
    return seed_rl_dataset

def validate_seed_item(item):
    """
    Validate that a seed item has all required components for SWE-RL training.
    """
    # Check issue description
    if not item.get('issue') or len(item['issue'].strip()) < 10:
        return False
    
    # Check code context
    if not item.get('ctx') or len(item['ctx']) == 0:
        return False
    
    # Check oracle patches
    if not item.get('patchgt') or len(item['patchgt']) == 0:
        return False
    
    return True

def test_reward_function(item):
    """
    Test the reward calculation function on a seed item.
    This simulates how rewards will be calculated during training.
    """
    # Get oracle patch
    oracle_patch = next(iter(item['patchgt'].values()))
    
    # Simulate correct format
    mock_correct = oracle_patch
    similarity = difflib.SequenceMatcher(None, mock_correct, oracle_patch).ratio()
    
    return similarity

def create_huggingface_dataset(data, output_path):
    """
    Create and save a HuggingFace dataset from the seed data.
    """
    try:
        dataset = Dataset.from_list(data)
        
        # Create train/validation split
        train_size = int(len(dataset) * 0.9)
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        dataset_dict.save_to_disk(output_path / "hf_dataset")
        print(f"Saved HuggingFace dataset to {output_path / 'hf_dataset'}")
    except Exception as e:
        print(f"Error creating HuggingFace dataset: {e}")

def preview_seed_data(output_dir='seed_rl_dataset', num_samples=2):
    """
    Preview a few examples from the seed dataset to verify the format.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    data_file = project_root / output_dir / "seed_rl_dataset.json"
    
    if not data_file.exists():
        print(f"Seed dataset file not found: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset contains {len(data)} examples")
    print(f"Previewing {num_samples} examples:")
    
    for i, item in enumerate(data[:num_samples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Issue:\n{item['issue'][:200]}...")
        print(f"\nCode context: {len(item['ctx'])} files")
        
        for filename, content in list(item['ctx'].items())[:1]:
            print(f"\nFilename: {filename}")
            print(f"Content preview:\n{content[:200]}...")
        
        print(f"\nOracle patches: {len(item['patchgt'])} files")
        
        # Test reward calculation
        test_reward = test_reward_function(item)
        print(f"\nTest reward calculation: {test_reward:.4f}")

if __name__ == "__main__":
    # Process data
    process_pr_data_for_swerl()
    
    # Preview a few examples
    preview_seed_data()
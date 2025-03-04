"""
Enhanced script to collect Pull Requests for SWE-RL training.
Collects complete context, relevant files, and creates proper structure for GRPO.
"""

from github import Github
import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path
import base64
import time
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment variables
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your .env file")

# Initialize GitHub client
g = Github(github_token)

# Target repositories (expanded list of AI/ML libraries)
target_repos = [
    "langchain-ai/langchain",
    "huggingface/transformers",
    "microsoft/DeepSpeed",
    "lm-sys/FastChat",
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "openai/openai-python",
    "ray-project/ray",
    "pytorch/vision",
    "pytorch/audio",
    "deepmind/dm-haiku",
    "jax-ml/jax",
    "ggerganov/llama.cpp",
    "microsoft/onnxruntime",
    "ml-explore/mlx"
]

# List of SWE-bench repositories to exclude to prevent data contamination
SWEBENCH_REPOS = [
    "django/django",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "sympy/sympy",
    "matplotlib/matplotlib"
    # Add other SWE-bench repos here
]

def is_bug_fix(pr_title, pr_body, issue_title, issue_body):
    """
    Determine if a PR is likely a bug fix based on its title and description.
    The paper emphasizes selecting PRs that are bug fixes.
    """
    bug_keywords = [
        'fix', 'bug', 'issue', 'error', 'crash', 'exception', 'fault',
        'defect', 'problem', 'incorrect', 'wrong', 'broken', 'regression'
    ]
    
    text = " ".join([t for t in [pr_title, pr_body, issue_title, issue_body] if t])
    text = text.lower()
    
    for keyword in bug_keywords:
        if re.search(rf'\b{keyword}s?\b', text):
            return True
            
    return False

def get_relevant_unmodified_files(repo, pr, modified_files):
    """
    Identify relevant unmodified files that provide context for the PR.
    The paper emphasizes including relevant but unchanged files.
    """
    relevant_files = []
    
    # Get all imports from modified Python files
    all_imports = set()
    modified_file_paths = [f['filename'] for f in modified_files]
    
    for file_data in modified_files:
        if file_data['filename'].endswith('.py'):
            content = file_data['content_before']
            
            # Extract imports
            import_lines = re.findall(r'^(?:from|import)\s+([.\w]+)(?:\s+import\s+|$)', 
                                     content, re.MULTILINE)
            
            for imp in import_lines:
                all_imports.add(imp.split('.')[0])  # Get top-level module
    
    # Search for files that might be relevant
    # This is simplified - in a real scenario, you might need more complex heuristics
    for content_file in repo.get_contents("", ref=pr.base.sha):
        if content_file.type == "dir" and content_file.name in all_imports:
            try:
                # Find Python files in this directory
                for py_file in repo.get_contents(content_file.path, ref=pr.base.sha):
                    if py_file.type == "file" and py_file.name.endswith('.py'):
                        # Skip if this file was already modified
                        if py_file.path in modified_file_paths:
                            continue
                            
                        try:
                            relevant_files.append({
                                'filename': py_file.path,
                                'content': base64.b64decode(py_file.content).decode('utf-8'),
                                'is_modified': False
                            })
                        except Exception as e:
                            print(f"    Error decoding file {py_file.path}: {e}")
            except Exception as e:
                print(f"    Error accessing directory {content_file.path}: {e}")
    
    return relevant_files[:5]  # Limit to 5 most relevant files to avoid too much data

def collect_prs(repo_name, limit=100):
    """
    Collect pull requests from a GitHub repository.
    
    Args:
        repo_name (str): Repository name in format "owner/repo"
        limit (int): Maximum number of PRs to collect
        
    Returns:
        list: List of PR data dictionaries
    """
    print(f"Collecting PRs from {repo_name}...")
    
    # Skip SWE-bench repositories to prevent data contamination
    for swebench_repo in SWEBENCH_REPOS:
        if swebench_repo.lower() in repo_name.lower():
            print(f"Skipping {repo_name} as it appears to be part of SWE-bench")
            return []
    
    repo = g.get_repo(repo_name)
    prs = []
    
    try:
        # Get merged PRs, sorted by most recently updated
        for pr in repo.get_pulls(state='closed', sort='updated', direction='desc'):
            if len(prs) >= limit:
                break
                
            if not pr.merged:
                continue
                
            print(f"  Processing PR #{pr.number}: {pr.title}")
            
            # Get issue if linked
            issue = None
            issue_title = None
            issue_body = None
            
            if hasattr(pr, 'issue_url') and pr.issue_url:
                issue_number = pr.issue_url.split('/')[-1]
                try:
                    issue = repo.get_issue(int(issue_number))
                    issue_title = issue.title
                    issue_body = issue.body
                except Exception as e:
                    print(f"    Error fetching issue: {e}")
            
            # Check if this is a bug fix PR
            if not is_bug_fix(pr.title, pr.body, issue_title, issue_body):
                print(f"    Skipping PR (not identified as a bug fix)")
                continue
                
            # Get PR description and files
            files_data = []
            for file in pr.get_files():
                if file.filename.endswith(('.py', '.js', '.java', '.c', '.cpp', '.h', '.cs')):
                    try:
                        # Get file content before PR changes
                        content_before = None
                        try:
                            content_file = repo.get_contents(file.filename, ref=pr.base.sha)
                            content_before = base64.b64decode(content_file.content).decode('utf-8')
                        except Exception:
                            # File might be new in this PR
                            content_before = ""
                        
                        files_data.append({
                            'filename': file.filename,
                            'content_before': content_before,
                            'patch': file.patch,
                            'is_modified': True
                        })
                    except Exception as e:
                        print(f"    Error fetching file {file.filename}: {e}")
                        continue
            
            # Skip if no code files were modified
            if not files_data:
                print(f"    Skipped PR (no valid code files)")
                continue
                
            # Get relevant unmodified files to provide additional context
            relevant_unmodified = get_relevant_unmodified_files(repo, pr, files_data)
            
            # Combine all files (modified and relevant unmodified)
            all_files = files_data + relevant_unmodified
            
            # Create the final PR data structure
            pr_data = {
                'number': pr.number,
                'title': pr.title,
                'body': pr.body,
                'issue_title': issue_title,
                'issue_body': issue_body,
                'files': all_files,
                # Add the oracle patch - the actual changes made in the PR
                'oracle_patch': {f['filename']: f['patch'] for f in files_data if f['patch']}
            }
            
            # The paper emphasizes complete file contents and the oracle patch
            if pr_data['oracle_patch']:
                prs.append(pr_data)
                print(f"    Added PR with {len(files_data)} modified files and {len(relevant_unmodified)} relevant files")
            else:
                print(f"    Skipped PR (no valid patches)")
            
            # Respect GitHub API rate limits
            time.sleep(1)
    
    except Exception as e:
        print(f"Error collecting PRs from {repo_name}: {e}")
    
    return prs

def organize_for_grpo(prs_data):
    """
    Organize PR data for GRPO training.
    Creates the seed RL dataset structure as described in the paper.
    """
    seed_rl_dataset = []
    
    for pr in prs_data:
        # Skip PRs without issues as the paper focuses on issue-solving
        if not pr.get('issue_title') and not pr.get('issue_body'):
            continue
            
        # Build the problem statement from issue and PR info
        problem_statement = ""
        if pr.get('issue_title'):
            problem_statement += f"Issue: {pr['issue_title']}\n\n"
        if pr.get('issue_body'):
            problem_statement += f"{pr['issue_body']}\n\n"
        
        # Prepare code context - all files content (modified and relevant unmodified)
        code_context = {}
        for file in pr['files']:
            if file.get('is_modified', True):
                code_context[file['filename']] = file['content_before']
            else:
                code_context[file['filename']] = file['content']
        
        # Oracle patch - the actual changes made
        oracle_patch = pr['oracle_patch']
        
        # Only add if we have all the necessary components
        if problem_statement and code_context and oracle_patch:
            seed_rl_dataset.append({
                'issue': problem_statement,
                'ctx': code_context,
                'patchgt': oracle_patch,
                # Metadata for grouping during GRPO training
                'repo': pr.get('repo', ''),
                'pr_number': pr.get('number', 0)
            })
    
    return seed_rl_dataset

def main():
    """Main function to collect PRs from all target repositories"""
    # Get the project root directory (one level up if inside scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    # Create output directory in the project root
    raw_output_dir = project_root / "pr_data_raw"
    seed_output_dir = project_root / "pr_data_seed"
    
    raw_output_dir.mkdir(exist_ok=True)
    seed_output_dir.mkdir(exist_ok=True)
    
    all_prs = []
    
    for repo_name in target_repos:
        # Collect PRs
        prs = collect_prs(repo_name, limit=100)
        
        if not prs:
            print(f"No valid PRs found for {repo_name}")
            continue
        
        # Add repo name to each PR
        for pr in prs:
            pr['repo'] = repo_name
        
        all_prs.extend(prs)
        
        # Save raw data to disk
        repo_short = repo_name.split('/')[-1]
        output_file = raw_output_dir / f"{repo_short}_raw.json"
        
        with open(output_file, 'w') as f:
            json.dump(prs, f, indent=2)
        
        print(f"Saved {len(prs)} raw PRs to {output_file}")
    
    # Process all PRs into the seed RL dataset format for GRPO
    seed_rl_dataset = organize_for_grpo(all_prs)
    
    # Group data by repository for GRPO training
    grouped_data = defaultdict(list)
    for item in seed_rl_dataset:
        repo = item['repo']
        grouped_data[repo].append(item)
    
    # Save grouped data
    for repo, items in grouped_data.items():
        repo_short = repo.split('/')[-1]
        output_file = seed_output_dir / f"{repo_short}_seed.json"
        
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)
        
        print(f"Saved {len(items)} seed items for {repo} to {output_file}")
    
    # Also save the complete dataset
    with open(seed_output_dir / "complete_seed_dataset.json", 'w') as f:
        json.dump(seed_rl_dataset, f, indent=2)
    
    print(f"Saved complete seed dataset with {len(seed_rl_dataset)} items")
    print("Dataset is structured for GRPO training as described in the SWE-RL paper")

if __name__ == "__main__":
    main()
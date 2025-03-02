"""
Script to collect Pull Requests from specific GitHub repositories.
Uses GitHub API to fetch merged PRs with their context and file contents.
"""

from github import Github
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment variables
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your .env file")

# Initialize GitHub client
g = Github(github_token)

# Target repositories (AI/ML libraries)
target_repos = [
    "langchain-ai/langchain",
    "huggingface/transformers",
    "microsoft/DeepSpeed",
    "lm-sys/FastChat"
]

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
    repo = g.get_repo(repo_name)
    prs = []
    
    try:
        for pr in repo.get_pulls(state='closed', sort='updated', direction='desc')[:limit]:
            if not pr.merged:
                continue
                
            print(f"  Processing PR #{pr.number}: {pr.title}")
            
            # Get issue if linked
            issue = None
            if pr.issue_url:
                issue_number = pr.issue_url.split('/')[-1]
                try:
                    issue = repo.get_issue(int(issue_number))
                except Exception as e:
                    print(f"    Error fetching issue: {e}")
                    pass
                    
            # Get PR description and files
            files_data = []
            for file in pr.get_files():
                if file.filename.endswith('.py'):
                    try:
                        # Get file content before PR changes
                        content_before = repo.get_contents(
                            file.filename, 
                            ref=pr.base.sha
                        ).decoded_content.decode('utf-8')
                        
                        files_data.append({
                            'filename': file.filename,
                            'content_before': content_before,
                            'patch': file.patch
                        })
                    except Exception as e:
                        print(f"    Error fetching file {file.filename}: {e}")
                        continue
            
            if files_data:
                prs.append({
                    'number': pr.number,
                    'title': pr.title,
                    'body': pr.body,
                    'issue_title': issue.title if issue else None,
                    'issue_body': issue.body if issue else None,
                    'files': files_data
                })
                print(f"    Added PR with {len(files_data)} Python files")
            else:
                print(f"    Skipped PR (no valid Python files)")
    
    except Exception as e:
        print(f"Error collecting PRs from {repo_name}: {e}")
    
    return prs

def main():
    """Main function to collect PRs from all target repositories"""
    # Create output directory
    output_dir = Path("pr_data")
    output_dir.mkdir(exist_ok=True)
    
    for repo_name in target_repos:
        # Collect PRs
        prs = collect_prs(repo_name)
        
        if not prs:
            print(f"No valid PRs found for {repo_name}")
            continue
        
        # Save to disk
        repo_short = repo_name.split('/')[-1]
        output_file = output_dir / f"{repo_short}.json"
        
        with open(output_file, 'w') as f:
            json.dump(prs, f, indent=2)
        
        print(f"Saved {len(prs)} PRs to {output_file}")

if __name__ == "__main__":
    main()
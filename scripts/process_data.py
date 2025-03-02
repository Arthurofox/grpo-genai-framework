"""
Process raw PR data into a format suitable for RL training.
Converts GitHub PR patches to search/replace format.
"""

import json
import os
import re
from pathlib import Path
from datasets import Dataset

def patch_to_search_replace(patch):
    """Convert a diff patch to search/replace format"""
    search_replace_edits = []
    
    # Parse the unified diff
    lines = patch.split('\n')
    chunks = []
    current_chunk = {'old_start': 0, 'old_lines': [], 'new_lines': []}
    
    for line in lines:
        if line.startswith('@@'):
            # New chunk
            if current_chunk['old_lines'] or current_chunk['new_lines']:
                chunks.append(current_chunk)
            # Parse the @@ -a,b +c,d @@ line
            match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
            if match:
                old_start = int(match.group(1))
                current_chunk = {'old_start': old_start, 'old_lines': [], 'new_lines': []}
        elif line.startswith('-'):
            current_chunk['old_lines'].append(line[1:])
        elif line.startswith('+'):
            current_chunk['new_lines'].append(line[1:])
        elif line.startswith(' '):
            current_chunk['old_lines'].append(line[1:])
            current_chunk['new_lines'].append(line[1:])
    
    if current_chunk['old_lines'] or current_chunk['new_lines']:
        chunks.append(current_chunk)
    
    # Convert chunks to search/replace format
    for chunk in chunks:
        if chunk['old_lines'] and chunk['new_lines']:
            search_replace_edits.append({
                'search': '\n'.join(chunk['old_lines']),
                'replace': '\n'.join(chunk['new_lines'])
            })
    
    return search_replace_edits

def process_pr_data(pr_data_dir='pr_data'):
    """Process PR data to create a dataset for training"""
    processed_data = []
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    # Full path to PR data directory
    pr_data_path = project_root / pr_data_dir
    
    if not pr_data_path.exists():
        raise FileNotFoundError(f"PR data directory not found: {pr_data_path}")
    
    print(f"Processing PR data from {pr_data_path}")
    
    for filename in os.listdir(pr_data_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = pr_data_path / filename
        print(f"Processing {file_path}")
        
        with open(file_path, 'r') as f:
            try:
                prs = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")
                continue
        
        for pr in prs:
            # Combine issue and PR description
            problem_statement = ""
            if pr.get('issue_title'):
                problem_statement += f"Issue: {pr['issue_title']}\n\n"
            if pr.get('issue_body'):
                problem_statement += f"{pr['issue_body']}\n\n"
            problem_statement += f"PR: {pr['title']}\n\n"
            if pr.get('body'):
                problem_statement += pr['body']
                
            for file_data in pr.get('files', []):
                # For each file in the PR
                try:
                    filename = file_data['filename']
                    content = file_data['content_before']
                    patch = file_data['patch']
                    
                    # Convert patch to search/replace format
                    edits = patch_to_search_replace(patch)
                    
                    if edits:
                        processed_data.append({
                            'problem_statement': problem_statement,
                            'filename': filename,
                            'content': content,
                            'edits': edits
                        })
                except KeyError as e:
                    print(f"Missing key in file data: {e}")
                except Exception as e:
                    print(f"Error processing {filename if 'filename' in locals() else 'unknown file'}: {e}")
    
    print(f"Processed {len(processed_data)} examples")
    
    # Convert to HuggingFace dataset
    return Dataset.from_list(processed_data)

if __name__ == "__main__":
    # Process data and save to disk
    dataset = process_pr_data()
    output_path = "processed_pr_data"
    dataset.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")
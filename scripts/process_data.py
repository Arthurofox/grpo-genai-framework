import json
import os
import difflib
import re
from datasets import Dataset

def patch_to_search_replace(content, patch):
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
    processed_data = []
    
    for filename in os.listdir(pr_data_dir):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(pr_data_dir, filename), 'r') as f:
            prs = json.load(f)
        
        for pr in prs:
            # Combine issue and PR description
            problem_statement = ""
            if pr['issue_title']:
                problem_statement += f"Issue: {pr['issue_title']}\n\n"
            if pr['issue_body']:
                problem_statement += f"{pr['issue_body']}\n\n"
            problem_statement += f"PR: {pr['title']}\n\n"
            if pr['body']:
                problem_statement += pr['body']
                
            for file_data in pr['files']:
                # For each file in the PR
                filename = file_data['filename']
                content = file_data['content_before']
                patch = file_data['patch']
                
                try:
                    edits = patch_to_search_replace(content, patch)
                    
                    if edits:
                        processed_data.append({
                            'problem_statement': problem_statement,
                            'filename': filename,
                            'content': content,
                            'edits': edits
                        })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Convert to HuggingFace dataset
    return Dataset.from_list(processed_data)

# Process data
dataset = process_pr_data()
dataset.save_to_disk('processed_pr_data')
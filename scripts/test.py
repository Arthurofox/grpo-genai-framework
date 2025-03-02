import os
from github import Github
from dotenv import load_dotenv

load_dotenv()

# Get GitHub token
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GitHub token not found")

# Test connection
g = Github(github_token)
user = g.get_user()
print(f"Successfully authenticated as: {user.login}")
print(f"Rate limit: {g.get_rate_limit().core.remaining}/{g.get_rate_limit().core.limit} requests remaining")
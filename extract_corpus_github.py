from github import Github
from dotenv import load_dotenv
import os
import json
load_dotenv()

GITHUB_ACCESS_TOKEN=os.getenv("GITHUB_ACCESS_TOKEN")
g = Github(GITHUB_ACCESS_TOKEN)

def get_repo_data(repo):
    # Basic repository info
    repo_data = {
        'name': repo.name,
        'description': repo.description,
        'url': repo.html_url,
        'branches': [],
        'issues': []  # Add an empty list for issues
    }

    # Fetch and add issues to repo_data
    for issue in repo.get_issues(state='all'):  # Fetch all issues, both open and closed
        issue_data = {
            'title': issue.title,
            'number': issue.number,
            'state': issue.state,
            'created_at': issue.created_at.isoformat(),
            'updated_at': issue.updated_at.isoformat(),
            'body': issue.body,
            'url': issue.html_url
        }
        repo_data['issues'].append(issue_data)

    # Branches and commits
    for branch in repo.get_branches():
        branch_data = {
            'name': branch.name,
            'commits': []
        }
        
        # List commits for each branch - limit set for demonstration purposes
        for commit in repo.get_commits(sha=branch.commit.sha):  # Adjust as needed
            commit_data = {
                'sha': commit.sha,
                'message': commit.commit.message,
                'date': commit.commit.author.date.isoformat(),
                'author': commit.commit.author.name,
                'url': commit.html_url
            }
            branch_data['commits'].append(commit_data)
        
        repo_data['branches'].append(branch_data)
    
    return repo_data


# Main function to get data for all repositories
def get_github_data():
    all_repos_data = []
    
    for repo in g.get_user().get_repos():
        repo_data = get_repo_data(repo)
        all_repos_data.append(repo_data)
    
    return all_repos_data

github_corpus = get_github_data()

# Save the corpus to a JSON file
with open('github_corpus.json', 'w',encoding="utf-8") as f:
    json.dump(github_corpus, f, indent=4)
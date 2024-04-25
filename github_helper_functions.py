import os
import json





def flatten_repo_data(all_repos_data):
    lines = [] 

    for repo_data in all_repos_data:
        repo_name = f"repo_name:{repo_data['name']}"
        repo_description = f"repo_description:{repo_data['description']}"

        # Handle issues for the repository
        for issue in repo_data['issues']:
            issue_body_cleaned = issue['body'].replace('\n', ' ')
            issue_line = [
                "Github:\n",
                "Issue:",
                repo_name,
                repo_description,
                f"issue_title:{issue['title']}",
                f"issue_number:{issue['number']}",
                f"issue_state:{issue['state']}",
                f"created_at:{issue['created_at']}",
                f"updated_at:{issue['updated_at']}",
                f"url:{issue['url']}",
            ]
            lines.append(','.join(issue_line))

        for branch in repo_data['branches']:
            branch_name = f"branch_name:{branch['name']}"
            for commit in branch['commits']:
                commit_message_cleaned = commit['message'].replace('\n', ' ')
                commit_line = [
                    "Github:\n",
                    "Commit:",
                    repo_name,
                    repo_description,
                    branch_name,
                    f"commit_message:{commit_message_cleaned}",
                    f"commit_date:{commit['date']}",
                    f"commit_author:{commit['author']}",
                    f"url:{commit['url']}"
                ]
                lines.append(','.join(commit_line))
    text = "\n".join(lines)
    return text

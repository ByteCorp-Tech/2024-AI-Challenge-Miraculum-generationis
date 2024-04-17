from jira import JIRA
import os
import json
from dotenv import load_dotenv

load_dotenv()


JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_INSTANCE_URL = os.getenv("JIRA_INSTANCE_URL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")


jira = JIRA(server=JIRA_INSTANCE_URL, basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN))

def create_corpus():
    corpus = []
    projects = jira.projects()
    for project in projects:
        project_data = {
            "project_key": project.key,
            "project_name": project.name,
            "issues": []
        }
        issues = jira.search_issues(f'project={project.key}', maxResults=100)
        for issue in issues:
            issue_data = {
                "issue_key": issue.key,
                "issue_summary": issue.fields.summary,
                "issue_type": issue.fields.issuetype.name,
                "issue_status": issue.fields.status.name,
                "comments": []
            }
            comments = jira.comments(issue)
            for comment in comments:
                comment_data = {
                    "comment_id": comment.id,
                    "comment_author": comment.author.displayName,
                    "comment_body": comment.body
                }
                issue_data["comments"].append(comment_data)
            project_data["issues"].append(issue_data)
        corpus.append(project_data)
    return corpus

if __name__ == "__main__":
    corpus = create_corpus()
    with open('corpus/jira_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
    print("Corpus has been extracted and saved.")

import os
import json




def flatten_corpus(corpus):
    texts = ''
    for project in corpus:
        for issue in project['issues']:
            texts += f"Issue/Ticket: Project:{project['project_name']}, Issue Key: {issue['issue_key']}, Summary: {issue['issue_summary']}, Type: {issue['issue_type']}, Status: {issue['issue_status']}\n"
            for comment in issue['comments']:
                texts += f"Comment: Comment ID: {comment['comment_id']}, Author: {comment['comment_author']}, Body: {comment['comment_body']}, Issue Key: {issue['issue_key']}\n"
    return texts






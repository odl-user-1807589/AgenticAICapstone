#!/bin/bash
# Usage: ./push_to_github.sh "Your commit message"

COMMIT_MSG=${1:-"Auto-push: Approved code"}

git add .
git commit -m "$COMMIT_MSG"
git push origin main
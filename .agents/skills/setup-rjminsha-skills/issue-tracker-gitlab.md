# Issue tracker: GitLab

Issues and PRDs for this repo live as GitLab issues. Use the `glab` CLI for all operations.

## Conventions

- **Create an issue**: `glab issue create --title "..." --description "..."`. Use a heredoc for multi-line descriptions.
- **Read an issue**: `glab issue view <number>`.
- **List issues**: `glab issue list --state opened` with appropriate `--label` filters.
- **Comment on an issue**: `glab issue note <number> --message "..."`
- **Apply labels**: `glab issue update <number> --label "..."`
- **Close**: `glab issue close <number>`

Infer the repo from `git remote -v` — `glab` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitLab issue.

## When a skill says "fetch the relevant ticket"

Run `glab issue view <number>`.

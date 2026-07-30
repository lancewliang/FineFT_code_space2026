---
name: context7
description: Search and fetch up-to-date framework, SDK, library documentation and code examples using Context7 CLI (ctx7) or Context7 MCP server. Use when needing accurate API signatures, code examples, or library documentation for React, PyTorch, FastAPI, Next.js, etc.
metadata:
  short-description: Fetch up-to-date documentation using Context7
---

# Context7 Documentation

Context7 provides up-to-date documentation, API references, and code examples for popular open-source packages, libraries, and frameworks.

## Usage Options

### 1. Via MCP Server
When Context7 MCP server is configured in `~/.codex/config.toml`:
- Use `resolve-library-id` with `libraryName` (and optional `query`) to find the canonical library ID.
- Use `query-docs` with `libraryId` and `query` to fetch relevant documentation snippets and code examples.

### 2. Via Context7 CLI (`ctx7`)
When using CLI commands:
- Search library ID: `npx -y ctx7 library <package-name> "<query>"`
- Query documentation: `npx -y ctx7 docs <library-id> "<query>"`

## Examples
- Search React library ID:
  `npx -y ctx7 library react "hooks"` -> returns `/react/react`
- Query React documentation:
  `npx -y ctx7 docs /react/react "useEffect cleanup"`
- Search FastAPI library ID:
  `npx -y ctx7 library fastapi "async endpoint"` -> returns `/tiangolo/fastapi`
- Query FastAPI documentation:
  `npx -y ctx7 docs /tiangolo/fastapi "background tasks"`

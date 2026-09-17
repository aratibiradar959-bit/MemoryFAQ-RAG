# Contributing to MemoryFAQ-RAG

Thanks for taking the time to contribute.

## Getting started

1. Fork the repository and create a branch from `master`.
2. Follow the setup instructions in the [README](README.md).
3. Make one focused change per pull request.
4. Run the relevant tests before submitting your work.

```bash
pytest
```

## Pull requests

- Use a concise, descriptive title.
- Explain what changed and why.
- Include tests for behavior changes when practical.
- Do not commit secrets, API keys, local Qdrant data, or `.env` files.
- Keep formatting and naming consistent with the surrounding code.

## Reporting issues

Use a clear title and include:

- Steps to reproduce the problem
- Expected and actual behavior
- Python version, operating system, and relevant error output

For security-sensitive reports, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

# Dependency Policy

- Root `requirements.txt` is kept for backward compatibility with the older workflow.
- `backend/requirements.txt` is the starting point for backend work.
- `ml/requirements.txt` is the starting point for model training and evaluation work.

Recommended usage:

- Backend team: install from `backend/requirements.txt`
- ML team: install from `ml/requirements.txt`
- Legacy code: use the root `requirements.txt` only when working on older paths

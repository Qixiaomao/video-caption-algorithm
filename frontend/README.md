Frontend layer for Chainlit UI.

Rules:
- Do not import torch or model code here.
- Communicate with FastAPI through REST only.
- Recommended command:
  chainlit run frontend/chainlit_app.py -w --host 127.0.0.1 --port 8000

# AI Climbing Coach — Phase06 Step 6.1

React + TypeScript + Tailwind v4 + Vite frontend and FastAPI backend.

## Setup (Windows PowerShell)

From the `phase06` directory:

### Backend terminal
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```
If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m uvicorn main:app --reload` without activating.

API: http://127.0.0.1:8000/api/health
Docs: http://127.0.0.1:8000/docs

### Frontend terminal
```powershell
cd frontend
npm install
npm run dev
```

Dashboard: http://127.0.0.1:5173

Node.js 20.19+ or 22.12+ recommended for the Vite toolchain. Python 3.10+.

## Behavior
- Select a video for local browser preview only (no upload).
- Example movement metrics and coaching are explicitly sample data and do not update when a video is selected.
- Frontend fetches `/api/health`, proxied to FastAPI on port 8000.
- No MediaPipe or Gemini execution and no API billing in this phase.

## Next steps
6.2 upload endpoint and file validation; 6.3 read existing analysis JSON; 6.4 synchronized movement visualization; 6.5 coaching evidence UI; 6.6 explicit processing and generation actions.

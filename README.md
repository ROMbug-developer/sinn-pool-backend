# SINN Pool Backend

FastAPI backend for SINN mining pool.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/submit_share` | Submit mining share |
| GET | `/stats/{wallet}` | Get wallet stats |
| GET | `/pool/stats` | Get pool totals |
| POST | `/admin/reset_monthly` | Reset monthly stats |

## Deploy to Render

### Option 1: Blueprint (Recommended)
1. Push this folder to a GitHub repo
2. Go to render.com → New → Blueprint
3. Connect your repo
4. Render auto-creates web service + PostgreSQL

### Option 2: Manual
1. Go to render.com → New → PostgreSQL
   - Name: `sinn-pool-db`
   - Plan: Free
   - Create

2. Go to render.com → New → Web Service
   - Connect GitHub repo (or use "Deploy from Git")
   - Name: `sinn-pool-api`
   - Runtime: Python 3
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   
3. Add Environment Variables:
   - `DATABASE_URL` → Copy from PostgreSQL "Internal Connection String"
   - `ADMIN_KEY` → Generate a random string

## Test Locally

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://localhost/sinn_pool"
uvicorn main:app --reload
```

Open http://localhost:8000/docs for Swagger UI.

## Submit Share Example

```bash
curl -X POST https://your-app.onrender.com/submit_share \
  -H "Content-Type: application/json" \
  -d '{
    "wallet": "0x1234567890abcdef1234567890abcdef12345678",
    "board_id": 1,
    "nonce": 12345678,
    "hash": "000000ff12345678901234567890123456789012345678901234567890123456",
    "difficulty": 4096
  }'
```

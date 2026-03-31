# DAC HealthPrice — Monorepo

Dynamic insurance pricing platform for Cambodia & Vietnam.

## Structure

```
may_mac02/
├── backend/     # FastAPI pricing engine (deploy to Render)
└── frontend/    # React wizard UI (deploy to Vercel)
```

## Backend
- FastAPI + PostgreSQL (Supabase)
- Poisson (frequency) + GBR (severity) ML models
- Cloudflare Worker proxy for security

## Frontend
- React + Vite
- 5-step pricing wizard
- AI advisor chat (Claude)
- Cloudflare Worker as API proxy

## Security
- All API calls go through Cloudflare Worker
- CF secret token enforced on Render
- CSP, HSTS, X-Frame-Options via vercel.json
- Rate limiting, body size limit, admin key protection

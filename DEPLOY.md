# Deploying RAG Q&A Agent to Railway

## Prerequisites
- A [Railway](https://railway.app) account (free tier works)
- Your `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)
- [Railway CLI](https://docs.railway.app/develop/cli) (optional but handy)

---

## Option A — Deploy via Railway Dashboard (easiest)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "RAG Q&A Agent v3"
   gh repo create rag-qa-agent --private --push --source=.
   ```

2. **Create a new Railway project**
   - Go to [railway.app/new](https://railway.app/new)
   - Choose **"Deploy from GitHub repo"**
   - Select your `rag-qa-agent` repository

3. **Add environment variables** (Settings → Variables)
   ```
   GROQ_API_KEY=gsk_your_key_here
   LLM_MODEL=llama3-8b-8192
   LLM_TEMPERATURE=0.0
   CHUNK_SIZE=512
   CHUNK_OVERLAP=64
   RETRIEVER_TOP_K=4
   MEMORY_WINDOW_K=5
   SESSION_TTL_SECONDS=3600
   MAX_SESSIONS=100
   ENABLE_INCREMENTAL_INDEX=true
   LOG_LEVEL=INFO
   ```
   > ⚠️ Do NOT set `APP_PORT` — Railway manages `$PORT` automatically.

4. **Add a Volume** for persistence (Settings → Volumes)
   - Mount path: `/app/data`
   - This persists your FAISS index and uploaded docs across deploys

5. **Deploy** — Railway auto-detects the Dockerfile and builds

6. **Get your URL** — Railway assigns a public URL like `https://rag-qa-agent-production.up.railway.app`

---

## Option B — Railway CLI (one-command)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project in repo root
railway init

# Link or create project
railway link   # pick existing, or:
# railway new

# Set secrets
railway variables set GROQ_API_KEY=gsk_your_key_here
railway variables set LLM_MODEL=llama3-8b-8192
railway variables set LLM_TEMPERATURE=0.0
railway variables set CHUNK_SIZE=512
railway variables set CHUNK_OVERLAP=64
railway variables set RETRIEVER_TOP_K=4
railway variables set MEMORY_WINDOW_K=5

# Deploy
railway up

# Open in browser
railway open
```

---

## Persistent Storage on Railway

Railway volumes ensure your uploaded documents and FAISS index survive redeploys:

```
Settings → Volumes → Add Volume
  Name: rag-data
  Mount path: /app/data
```

Without a volume, the index is lost on every deploy and you'll need to re-upload and re-ingest.

---

## Free Tier Limits

| Resource | Railway Free |
|----------|-------------|
| RAM | 512 MB |
| CPU | 0.5 vCPU |
| Sleep | After 30 min inactivity |
| Bandwidth | 100 GB/month |

**Tips for free tier:**
- Use `llama3-8b-8192` (faster, less memory pressure)
- Keep `RETRIEVER_TOP_K=4` or lower
- Avoid very large PDFs on free tier (embedding is CPU-intensive)

For always-on production, upgrade to Railway's Hobby plan ($5/month).

---

## Custom Domain

1. Railway Dashboard → Settings → Networking → Custom Domain
2. Add your domain (e.g. `rag.yourdomain.com`)
3. Point a CNAME at the Railway-provided hostname

---

## Troubleshooting

**Deploy fails with "No start command"**
→ Ensure `railway.toml` is in the repo root (it's included in this project).

**Index not persisting across deploys**
→ You haven't added a Volume. See "Persistent Storage" above.

**Out of memory during ingest**
→ Upgrade Railway plan, or reduce `CHUNK_SIZE` and number of docs.

**Cold start is slow**
→ The embedding model (`all-MiniLM-L6-v2`) downloads on first boot. Subsequent starts use the Railway layer cache and are faster.

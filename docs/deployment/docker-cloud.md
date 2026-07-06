# Docker Cloud and Multi-Client Deployment

Docker is reserved for later cloud, server, or LAN multi-client deployments.
Local single-user Windows installs should use the portable `LocalMathRAG.exe`
release package instead.

## Intended Use

- Run the WebApp on a server reachable by multiple browser clients.
- Persist knowledge bases and model files with Docker volumes.
- Use llama.cpp as a local container service, or point the WebApp at another
  OpenAI-compatible endpoint.
- Keep public internet search, cloud model calls, and remote RAG services out of
  the default deployment unless explicitly configured by the operator.

## Local Compose Smoke Test

```powershell
docker compose -f docker-compose.local.yml config
```

## Server Startup

CPU:

```powershell
docker compose -f docker-compose.local.yml --profile llama-cpu up -d --build
```

CUDA:

```powershell
docker compose -f docker-compose.local.yml --profile llama-cuda up -d --build
```

WebApp only, with an external model endpoint configured in the UI:

```powershell
docker compose -f docker-compose.local.yml up -d --build webapp
```

## Ports and Volumes

- WebApp: `0.0.0.0:8765`
- llama.cpp OpenAI-compatible endpoint: `0.0.0.0:8080/v1`
- Persistent data: `./data:/app/data`
- Model mount: `./data/models:/models:ro`

For cloud use, put a reverse proxy and authentication layer in front of the
WebApp before exposing it beyond a trusted LAN.

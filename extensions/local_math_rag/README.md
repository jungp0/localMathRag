# LocalMathRAGFlow Extension

This directory contains the extension contract mounted into the RAGFlow Docker services.

It is intentionally small and stable:

- `schemas/` defines structured evidence objects.
- `config/pipeline.yaml` defines the parser/object extraction pipeline contract.
- `prompts/engineering_chat_system.md` defines the engineering chat tone.
- `api/openapi.yaml` defines sidecar endpoints for external agents and future RAGFlow integration.

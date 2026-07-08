from __future__ import annotations

from pathlib import Path


CONF_DIR = Path("/ragflow/conf")

LOCAL_EMBEDDING_BLOCK = """    embedding_model:
      name: 'Qwen/Qwen3-Embedding-0.6B'
      factory: 'OpenAI-API-Compatible'
      api_key: 'local'
      base_url: 'http://localmathrag-object-service:8088/v1'
    rerank_model:
      name: 'BAAI/bge-reranker-v2-m3'
      factory: 'OpenAI-API-Compatible'
      api_key: 'local'
      base_url: 'http://localmathrag-object-service:8088/v1'"""

UPSTREAM_EMBEDDING_BLOCKS = (
    """    embedding_model:
      api_key: 'xxx'
      base_url: 'http://tei:80'""",
    """    embedding_model:
      api_key: 'xxx'
      base_url: 'http://${TEI_HOST}:80'""",
)


def patch_config(path: Path) -> None:
    text = path.read_text()
    for block in UPSTREAM_EMBEDDING_BLOCKS:
        text = text.replace(block, LOCAL_EMBEDDING_BLOCK)
    if "rerank_model:" not in text and "base_url: 'http://localmathrag-object-service:8088/v1'" in text:
        text = text.replace(
            "      base_url: 'http://localmathrag-object-service:8088/v1'",
            "      base_url: 'http://localmathrag-object-service:8088/v1'\n"
            "    rerank_model:\n"
            "      name: 'BAAI/bge-reranker-v2-m3'\n"
            "      factory: 'OpenAI-API-Compatible'\n"
            "      api_key: 'local'\n"
            "      base_url: 'http://localmathrag-object-service:8088/v1'",
            1,
        )
    path.write_text(text)


for config_name in ("service_conf.yaml", "service_conf.yaml.template"):
    config_path = CONF_DIR / config_name
    if config_path.exists():
        patch_config(config_path)

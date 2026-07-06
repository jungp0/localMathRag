FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.116.1 uvicorn[standard]==0.35.0

COPY services/object_service/main.py /app/main.py
COPY extensions/local_math_rag/schemas /schemas

ENV LOCALMATHRAG_SCHEMA_DIR=/schemas
EXPOSE 8088

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]

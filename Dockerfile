FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8765

CMD ["python", "-m", "lookup_tool.cli", "serve", "--host", "0.0.0.0", "--port", "8765"]

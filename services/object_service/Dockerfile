FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
      fastapi==0.116.1 \
      uvicorn[standard]==0.35.0 \
      numpy==1.26.4 \
      onnxruntime==1.27.0 \
      tokenizers==0.23.1 \
      opencv-python-headless==4.11.0.86 \
      Pillow==12.3.0 \
      PyYAML==6.0.3 \
      chardet==7.4.3 \
    && pip install --no-cache-dir --no-deps rapid_latex_ocr==0.0.9

RUN pip install --no-cache-dir requests==2.32.5

COPY services/object_service/main.py /app/main.py
COPY services/object_service/localmathrag_math_ocr.py /app/localmathrag_math_ocr.py
COPY extensions/local_math_rag/schemas /schemas

ENV LOCALMATHRAG_SCHEMA_DIR=/schemas
EXPOSE 8088

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]

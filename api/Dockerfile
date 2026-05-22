FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


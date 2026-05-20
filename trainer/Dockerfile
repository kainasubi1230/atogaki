FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /tmp/api-requirements.txt
COPY trainer/requirements.txt /tmp/trainer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt -r /tmp/trainer-requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY . /app
ENV PYTHONPATH=/app

CMD ["python", "trainer/trainer_worker.py"]


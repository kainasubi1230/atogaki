FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /tmp/api-requirements.txt
COPY trainer/requirements.txt /tmp/trainer-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt -r /tmp/trainer-requirements.txt

COPY . /app
ENV PYTHONPATH=/app

CMD ["python", "trainer/trainer_worker.py"]


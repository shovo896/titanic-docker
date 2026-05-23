FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt
COPY train.py .
EXPOSE 8000
CMD ["python", "train.py"]


FROM python:3.11-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and trained model
COPY main.py .
COPY house_rent_model.joblib .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Golden Fade AI Style Preview backend
# Deploy this as a free Hugging Face "Docker" Space (CPU basic tier is
# enough — this container just proxies to the GPU Space, it doesn't
# run the model itself). See README.md for step-by-step instructions.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Hugging Face Docker Spaces expect the app to listen on port 7860
EXPOSE 7860

CMD ["python", "app.py"]

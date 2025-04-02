FROM python:3.9-slim
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY Home.py .
COPY yolo_predictions.py .
COPY models/ models/
COPY pages/ pages/
COPY images/ images/

EXPOSE 8501

CMD ["streamlit", "run", "Home.py"]
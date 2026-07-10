# Guardini — Forecasting NeuralProphet (Streamlit)
FROM python:3.10-slim

WORKDIR /app

# libgomp1: richiesta a runtime da torch/scipy per il multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU-only: il server Hetzner non ha GPU. Installandolo prima si evita
# che neuralprophet trascini la build CUDA di default (~2 GB in più, inutile
# e potenziale causa di crash per esaurimento memoria su VM piccole).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY guardini_j_galileo_v8.py .
COPY guardini.png .
COPY modelli/ ./modelli/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "guardini_j_galileo_v8.py", "--server.port=8501", "--server.address=0.0.0.0"]

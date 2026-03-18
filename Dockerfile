FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default

# Supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Application code
COPY production/ production/
COPY streamlit_app.py .
COPY warranty_samples.csv .

# Model artifacts
COPY models/ models/

EXPOSE 8080

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

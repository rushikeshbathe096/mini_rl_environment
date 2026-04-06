FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

COPY . .

ENV PORT=7860
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

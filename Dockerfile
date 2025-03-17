# Use the official Python 3.9.6 slim image
FROM python:3.9.6-slim-buster

# PDM version
ENV PDM_VERSION=2.22.3

# Set the PORT environment variable to Cloud Run's default
ENV PORT=8080

WORKDIR /code

# Copy dependency files
COPY pdm.lock pyproject.toml ./
RUN python -m pip install --upgrade pip --no-cache-dir && \
    pip install pdm==${PDM_VERSION} --no-cache-dir && \
    pdm install --prod --no-lock --no-editable

# Copy application files
COPY ./app /code/app
COPY ./model /code/model

# Set PATH to include PDM's virtual environment
ENV PATH="/code/.venv/bin:$PATH"

# Expose the port (optional, for documentation)
EXPOSE ${PORT}

# Run Gunicorn with shell form
CMD gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --workers 4 --timeout 120 app.main:app
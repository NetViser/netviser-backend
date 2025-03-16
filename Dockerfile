# Use the official Python 3.9.6 slim image (Debian-based slim version)
FROM python:3.9.6-slim-buster

# PDM version
ENV PDM_VERSION=2.22.3

# Set the PORT environment variable to Cloud Run's default
ENV PORT=8080

WORKDIR /code

# Copy only dependency management files to leverage Docker caching
COPY pdm.lock pyproject.toml ./
RUN python -m pip install --upgrade pip
RUN pip install pdm==${PDM_VERSION} && pdm install --prod --no-lock --no-editable

# Copy the rest of the files
COPY ./app /code/app
COPY ./model /code/model

# Expose the port (optional, for documentation; Cloud Run ignores this)
EXPOSE ${PORT}

# Run the application using shell form to substitute $PORT
CMD pdm run uvicorn app.main:app --host 0.0.0.0 --port $PORT
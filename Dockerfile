# Use the official Python 3.9.6 slim image (using Debian-based slim version)
FROM python:3.9.6-slim-buster

# PDM version
ENV PDM_VERSION=2.22.3

WORKDIR /code

# Copy only the dependency management files to leverage Docker caching.
COPY pdm.lock pyproject.toml ./
RUN python -m pip install --upgrade pip
RUN pip install pdm==${PDM_VERSION} && pdm install --prod --no-lock --no-editable

# Copy the rest of the files
COPY ./app /code/app
COPY ./model /code/model

# Expose the port
EXPOSE 8000

# Run the application
CMD ["pdm", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
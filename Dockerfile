# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install SQLite system dependencies
RUN apt-get update && apt-get install -y libsqlite3-dev

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY ./mom_service ./mom_service
COPY config.yaml .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONPATH=/app

# Run main.py when the container launches
# Includes --reload and --reload-include to watch config.yaml for changes.
# This is useful for development; for production, you might remove these flags.
CMD ["uvicorn", "mom_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-include", "config.yaml"]

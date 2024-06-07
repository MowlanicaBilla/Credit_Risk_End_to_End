# Use the official Python image with version 3.11.6 as base
FROM python:3.11.6-slim

# Install build dependencies
RUN apt-get update && apt-get install -y gcc python3-dev

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "--server.port", "8501", "streamlit_application.py"]
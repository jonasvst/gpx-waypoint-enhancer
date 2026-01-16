# 1. Use an official Python image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies for maps/data
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements and install them
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 5. Copy the rest of your app code
COPY . .

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. Start the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use a lightweight Python version
FROM python:3.11-slim

# Set the working directory in the server
WORKDIR /app

# Copy the dependency file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Tell the server to listen on port 8501 (Streamlit's default)
EXPOSE 8501

# Command to launch the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

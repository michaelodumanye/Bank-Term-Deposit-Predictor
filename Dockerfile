# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt (make sure you have this file with your needed packages)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project files into the container
COPY . .

# Expose port 8501 (Streamlit default)
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

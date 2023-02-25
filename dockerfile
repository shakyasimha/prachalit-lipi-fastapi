FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents to the directory at /app
COPY . /app

# Installing the dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Expose port 5000 for the application
EXPOSE 5000

# Start the server  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
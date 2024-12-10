FROM python:3.9-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Flask environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port and run Flask
EXPOSE 8080
CMD ["python", "app.py"]

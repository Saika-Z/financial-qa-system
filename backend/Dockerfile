# use lightweight python
FROM python:3.9-slim

# set working directory
WORKDIR /app

# set env variables to prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update && \ apt-get install -y -no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. copy dependencies, reduce build time
COPY requirements.txt .
RUN pip -install --no-cache-dir -r requirements.txt

# 2. copy code
COPY . .

# 3. expose port
EXPOSE 8000

# 4. run
CMD ["python", "main.py"]
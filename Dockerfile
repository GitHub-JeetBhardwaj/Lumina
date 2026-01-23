# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . .

# Create the cache directory for Torch (avoids permission errors)
RUN mkdir -p /.cache
RUN chmod 777 /.cache

# Open port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py"]
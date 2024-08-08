# Contains python3.10
FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /app

# Install the Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app
CMD ["python", "cases/mnist/butterfly_classification.py"]

# Contains python3.8
FROM nvcr.io/nvidia/pytorch:21.10-py3

WORKDIR /app

# Install the Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# RUN pip install opencv-python==4.8.0.74

# Copy the rest of the application code into the container
COPY . /app
#CMD ["python", "cases/mnist/mnist_classification.py"]

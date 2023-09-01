# syntax=docker/dockerfile:experimental
FROM ubuntu:latest

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \ 
    libglib2.0-0 \   
    && apt-get install -y libssl-dev \  
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set LD_LIBRARY_PATH environment variable
# ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

# Copy and install Unstructured
COPY requirements requirements

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache -r requirements/base.txt && \
  pip3 install unstructured_paddleocr

COPY unstructured_inference unstructured_inference
COPY test.py test.py

CMD ["/bin/bash"]

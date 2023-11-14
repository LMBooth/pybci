
# Use Ubuntu 20.04 LTS as the base image
FROM ubuntu:22.04

# Avoid prompts from apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install Python 3.10
RUN apt-get install -y python3.10 python3-distutils python3-pip

# Install additional dependencies
RUN apt-get install -y cmake git libpugixml1v5 wget

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    /usr/bin/python3.10 get-pip.py && \
    rm get-pip.py

# Clone the pybci repository
RUN git clone https://github.com/LMBooth/pybci.git /pybci

# Set the working directory
WORKDIR /pybci

# Install Python dependencies
RUN /usr/bin/python3.10 -m pip install --upgrade urllib3>=2.0.5
RUN /usr/bin/python3.10 -m pip install . pytest pytest-timeout ruff

# Download and install liblsl
RUN wget https://github.com/sccn/liblsl/releases/download/v1.16.2/liblsl-1.16.2-focal_amd64.deb -O liblsl.deb && \
    dpkg -i liblsl.deb && \
    rm liblsl.deb

# Copy liblsl.so to the target directory
RUN mkdir -p /home/appveyor/.local/lib/python3.10/site-packages/pylsl/lib && \
    cp /usr/lib/liblsl.so /home/appveyor/.local/lib/python3.10/site-packages/pylsl/lib/

# Expose the necessary port (change if needed)
EXPOSE 8080

# Command to run when starting the container
CMD ["/usr/bin/python3.10", "-m", "pybci"]

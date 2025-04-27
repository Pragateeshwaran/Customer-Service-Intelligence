# Use base image with CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Disable interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set work directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install streamlit
RUN pip install speechbrain
RUN pip install transformers
RUN pip install sentencepiece
RUN pip install langchain langchain_groq groq

# Expose Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "gui_final.py", "--server.port=8501", "--server.address=0.0.0.0"]

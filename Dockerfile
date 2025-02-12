# Use the official AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Set HOME to /tmp to ensure writable directories
ENV HOME=/tmp

# Install system dependencies required by PaddleOCR and other packages
RUN yum update -y && \
    yum install -y wget tar git gcc-c++ libglib2.0 libSM libXrender libXext mesa-libGL xz && \
    yum clean all

# Install ffmpeg via static build
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xvf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-static

# Verify ffmpeg installation
RUN ffmpeg -version

# Upgrade pip, setuptools, and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Install PaddlePaddle (CPU version)
RUN pip3 install paddlepaddle==2.4.2

# Install Python dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Install yt-dlp globally so "yt-dlp" is on the PATH
RUN pip3 install --no-cache-dir yt-dlp

# Create custom directories within /var/task for PaddleOCR models
RUN mkdir -p /var/task/paddleocr_models/det && \
    mkdir -p /var/task/paddleocr_models/rec && \
    mkdir -p /var/task/paddleocr_models/cls

# Download and extract PaddleOCR detection model
RUN wget -O /var/task/paddleocr_models/det/en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && \
    tar -xvf /var/task/paddleocr_models/det/en_PP-OCRv3_det_infer.tar -C /var/task/paddleocr_models/det && \
    rm /var/task/paddleocr_models/det/en_PP-OCRv3_det_infer.tar

# Download and extract PaddleOCR recognition model
RUN wget -O /var/task/paddleocr_models/rec/en_PP-OCRv3_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar && \
    tar -xvf /var/task/paddleocr_models/rec/en_PP-OCRv3_rec_infer.tar -C /var/task/paddleocr_models/rec && \
    rm /var/task/paddleocr_models/rec/en_PP-OCRv3_rec_infer.tar

# Download and extract PaddleOCR classifier model
RUN wget -O /var/task/paddleocr_models/cls/ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && \
    tar -xvf /var/task/paddleocr_models/cls/ch_ppocr_mobile_v2.0_cls_infer.tar -C /var/task/paddleocr_models/cls && \
    rm /var/task/paddleocr_models/cls/ch_ppocr_mobile_v2.0_cls_infer.tar

COPY cookies.txt /var/task/cookies.txt

# Copy the Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Define the entrypoint and command to run the Lambda function
CMD ["lambda_function.lambda_handler"]

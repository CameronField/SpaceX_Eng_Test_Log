# Use the AWS Lambda Python 3.13 base image
FROM public.ecr.aws/lambda/python:3.13

# Install system dependencies using microdnf (git is needed for cloning)
RUN microdnf update -y && \
    microdnf install -y git && \
    microdnf clean all

# Set MPLCONFIGDIR to a writable directory to avoid matplotlib warnings
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the Lambda function code
COPY lambda_function.py .

# Set the CMD to point to your handler (module:function format)
CMD [ "lambda_function.lambda_handler" ]

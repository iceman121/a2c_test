# OS and interpreter setup
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# WORKDIR /opt

# Install requirements
# COPY ./requirements.txt /opt/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# RUN apt-get install -y python-opengl > /dev/null 2>&1
# RUN pip install git+https://github.com/tensorflow/docs > /dev/null 2>&1

# Execution
# RUN chmod +x ./main.py
# CMD ["./main.py"]

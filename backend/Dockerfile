FROM ubuntu:22.04

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 验证安装的Python版本
RUN python3.10 --version && pip3 --version

# 将python3.10设置为默认的python版本（可选）
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --set python /usr/bin/python3.10

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

COPY . /app

RUN mkdir /app/Databases
RUN mkdir /app/Logs
RUN mkdir /app/static
RUN mkdir /app/static/avatar
RUN mkdir /app/static/uploads
RUN mkdir /app/Database_global
ENV PYTHONPATH "${PYTHONPATH}:/app"
CMD ["python", "main.py"]

# C++ opponent panel: official vendored engine + g++ to build competitors' .so
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --disable-pip-version-check jsonschema==4.23.0 requests==2.32.3
WORKDIR /work

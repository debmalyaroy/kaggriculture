# Linux harness image for the Track-P COMPILED agent.
#
# The compiled binary is Linux x86_64 only, so the artefact must be measured on
# Linux. This image carries nothing but CPython and the one dependency the
# VENDORED official interpreter actually needs (jsonschema, requests) -- the engine
# itself is bind-mounted from vendor/, never baked in, so the harness always
# plays the same interpreter the rest of the repo does.
#
#   docker build -f scripts/trackp_harness.Dockerfile -t kagg-harness:2 .
FROM python:3.11-slim
RUN pip install --no-cache-dir --disable-pip-version-check jsonschema==4.23.0 requests==2.32.3
WORKDIR /work

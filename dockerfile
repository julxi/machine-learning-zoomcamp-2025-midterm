FROM python:3.13.5-slim-bookworm

WORKDIR /code

# dependencies
RUN pip install uv
COPY ["pyproject.toml", "uv.lock", "./"]
RUN uv sync --no-dev

# pipeline and server
COPY ["pipeline_v1.bin", "./"]
COPY ["server.py", "./"]

# run app under 9696
EXPOSE 9696
ENTRYPOINT ["uv", "run", "--no-dev", "uvicorn", "server:app", "--host=0.0.0.0", "--port=9696"]

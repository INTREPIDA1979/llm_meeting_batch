FROM python:3.11-slim

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN set -ex; \
    pip install -r requirements.txt; \
    pip install gunicorn

COPY . /job
WORKDIR /job

ENV TZ=Asia/Tokyo

CMD ["python", "main.py"]


FROM python:3.10-bullseye

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       build-essential python3-dev git libssl-dev libffi-dev libgomp1 \
       libglib2.0-0 libgl1-mesa-glx curl wget ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser/app

COPY --chown=sandboxuser:sandboxuser backend/ ./backend/

COPY --chown=sandboxuser:sandboxuser requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


USER sandboxuser

# By default, run the validation script
ENTRYPOINT ["python3", "backend/libs/train_model.py"]
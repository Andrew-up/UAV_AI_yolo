FROM python:3.9.16-slim-bullseye

COPY requirements.txt .

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
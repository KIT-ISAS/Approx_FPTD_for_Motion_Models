FROM tensorflow/tensorflow:2.1.0-gpu-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
apt-get install -y --no-install-recommends python3.6-tk && \
pip3 install --upgrade pip && \ 
pip3 install matplotlib==3.2.1

RUN pip3 install --upgrade pip && \
pip3 install scikit-learn==0.22.2.post1 && \ 
pip3 install pandas==1.0.3

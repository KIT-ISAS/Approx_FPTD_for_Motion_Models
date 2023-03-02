FROM tensorflow/tensorflow:2.8.0-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
apt-get update && \
apt-get install -y --no-install-recommends python3.8-tk && \
pip3 install --upgrade pip && \ 
pip3 install matplotlib==3.2.1  # because newer versions may cause problems with display forwarding from gpu server

RUN pip3 install --upgrade pip && \
pip3 install --upgrade tensorflow-probability==0.16.0 && \
pip3 install scikit-learn==1.0.2 && \ 
pip3 install pandas==1.4.2 && \
pip3 install POT==0.8.2

# RUN apt-get install -y texlive-science=3.14159265 && \
RUN apt-get install -y texlive-full
# && \ apt-get install -y font-manager

RUN apt-get install -y msttcorefonts -qq && \
rm ~/.cache/matplotlib -rf
# RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
# apt-get install -y ttf-mscorefonts-installer && \

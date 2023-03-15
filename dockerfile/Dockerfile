FROM tensorflow/tensorflow:2.8.0-gpu

ENV DEBIAN_FRONTEND noninteractive

# Versions of already installed packages can be identified using: apt-cache policy <package name>

# matplotlib
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC \
&& apt-get update \
&& apt-get install -y --no-install-recommends python3.8-tk \
&& pip3 install --upgrade pip \
&& pip3 install matplotlib==3.7.0  # note that older versions, e.g., 3.2.1, may cause warning when use latex fonts, but
# newer versions may cause problems with display forwarding from gpu server, which works with 3.2.1 (thus, choose the
# version carefully).

# python packages
RUN pip3 install --upgrade pip \
&& pip3 install --upgrade tensorflow-probability==0.16.0 \
&& pip3 install scikit-learn==1.0.2 \
&& pip3 install pandas==1.4.2 \
&& pip3 install POT==0.8.2

# latex
RUN apt-get update \
&& apt-get install -y dvipng=1.15-1.1 \
# required for agg, see https://matplotlib.org/stable/tutorials/text/usetex.html
# && apt-get install -y texlive-full=2019.20200218-1 && \  # this works fine, but requires a lot of space and takes hours...
&& apt-get install -y texlive-latex-extra=2019.202000218-1 \
# install only required packages
&& apt-get install -y texlive-science=2019.202000218-1 \
&& apt-get install -y texlive-xetex=2019.20200218-1 \
&& apt-get install -y texlive-fonts-recommended=2019.20200218-1 \
&& apt-get install -y texlive-fonts-extra=2019.202000218-1 \
# use the base latex versions, manually install the necessary dependencies, see
# https://matplotlib.org/stable/devel/dependencies.html#optional-dependencies
&& apt-get install -y cm-super=0.3.4-15 \
# use the texlive package manager to install additional packages
&& tlmgr init-usertree \
# before doing so set up the user tree with the init-usertree action (because were are in user mode)
&& tlmgr repository add ftp://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2019/tlnet-final \
&& tlmgr repository remove http://mirror.ctan.org/systems/texlive/tlnet \
&& tlmgr option repository ftp://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2019/tlnet-final \
# the last three commands set up tlmgr to work with the 2019 texlive version, see
# https://tex.stackexchange.com/questions/540429/tlmgr-in-ubuntu-20-04-local-tex-live-2019-is-older-than-remote-repository-2
 && tlmgr --verify-repo=none install underscore amsmath
# install underscore and amsfont latex packages, underscore is a required dependecy, see
# https://matplotlib.org/stable/devel/dependencies.html#optional-dependencies
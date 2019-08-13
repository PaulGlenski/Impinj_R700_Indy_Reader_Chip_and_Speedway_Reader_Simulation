FROM fedora:27

WORKDIR /mnt/work

# For basic convenience:
RUN dnf -y install \
    binutils \
    curl \
    findutils \
    gzip \
    tar \
    unzip \
    wget \
    which

# For building/compiling:
RUN dnf -y install \
    automake \
    cmake \
    gcc \
    gcc-c++ \
    glibc \
    libtool \
    make

# For documentation:
RUN dnf -y install \
    python \
    python-devel \
    python-mtTkinter \
    python-sphinx \
    redhat-rpm-config

COPY requirements.txt /mnt/work/requirements.txt

RUN pip install -r /mnt/work/requirements.txt


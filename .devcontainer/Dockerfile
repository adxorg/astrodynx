FROM ghcr.io/nvidia/jax:jax-2025-06-01
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends pandoc

ARG USERNAME=ubuntu
RUN mkdir -p /home/$USERNAME/.pip \
    && echo '[global]' > /home/$USERNAME/.pip/pip.conf \
    && echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> /home/$USERNAME/.pip/pip.conf \
    && echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> /home/$USERNAME/.pip/pip.conf

ENV LC_ALL=C.UTF-8

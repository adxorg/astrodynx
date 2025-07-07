FROM ghcr.io/nvidia/jax:jax-2025-07-06
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources

ARG USERNAME=ubuntu
RUN mkdir -p /home/$USERNAME/.pip \
    && echo '[global]' > /home/$USERNAME/.pip/pip.conf \
    && echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> /home/$USERNAME/.pip/pip.conf \
    && echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> /home/$USERNAME/.pip/pip.conf

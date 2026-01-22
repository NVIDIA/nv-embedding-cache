FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install useful dev tools
RUN apt-get update && apt-get -y install sudo nano gdb time htop nvtop redis

# Install clang
ENV CLANG_VER=20
RUN wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh
RUN chmod a+x /tmp/llvm.sh
RUN apt update
RUN apt install -y lsb-release software-properties-common gnupg
RUN /tmp/llvm.sh ${CLANG_VER}
RUN apt install -y clang-tools-${CLANG_VER} clang-format-${CLANG_VER} clang-tidy-${CLANG_VER}

# Create user matching the host user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Relax image constraint for torch
RUN sed -i '/^torch==/c\torch>=2.7' /etc/pip/constraint.txt

USER $USERNAME

# Keep container running
CMD ["sleep", "infinity"]

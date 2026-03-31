FROM nvcr.io/nvidia/pytorch:26.01-py3
RUN apt-get update
RUN apt-get -y install sudo redis

ARG UID
ARG UNAME
ARG GID
ARG GNAME
RUN if [ -n "$UID" -a -n "$UNAME" -a -n "$GID" -a -n "$GNAME" ]; then \
    (groupadd -g $GID $GNAME || true) && useradd --uid $UID -g $GID --no-log-init --create-home $UNAME && (echo "${UNAME}:password" | chpasswd) && (echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers); \
    fi
USER $UNAME
ARG START_DIR=/home/${UNAME}
WORKDIR $START_DIR

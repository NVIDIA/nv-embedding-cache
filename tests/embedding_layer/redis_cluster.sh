#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
MAX_SLEEP=30
DEFAULT_PORT=6379
RUNTIME_ROOT=${NVE_REDIS_RUNTIME_DIR:-/dev/shm/nve-redis}

usage() {
  echo "Usage: $0 [start | start_cluster | start_single [port] | stop]"
  exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi

if [ "$#" -eq 2 ] && [ "$1" != "start_single" ]; then
  usage
fi

if [ "$1" = "start_single" ] || [ "$1" = "start" ]; then
  SINGLE_PORT=${2:-$DEFAULT_PORT}
  if ! [[ "$SINGLE_PORT" =~ ^[0-9]+$ ]] || [ "$SINGLE_PORT" -lt 1 ] || [ "$SINGLE_PORT" -gt 65535 ]; then
    printf "${RED}Invalid port '${SINGLE_PORT}'!${NC}\n"
    exit 1
  fi
fi

# Wait until the redis-server on the given port answers PONG.
# Returns non-zero if it never came up within MAX_SLEEP seconds.
wait_for_server() {
  local port=$1
  local waited=0
  while [ "$(redis-cli -p $port ping 2>/dev/null)" != "PONG" ] && [ ${waited} != ${MAX_SLEEP} ]; do
    ((waited+=1))
    sleep 1
  done
  [ $waited != $MAX_SLEEP ]
}

port_is_free() {
  local port=$1
  ! (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

require_port_free() {
  local port=$1
  if ! port_is_free "$port"; then
    printf "${RED}Port ${port} is already in use - aborting!${NC}\n"
    exit 3
  fi
}

server_count() {
  pgrep -x redis-server 2>/dev/null | wc -l
}

wait_for_servers_to_stop() {
  local waited=0
  while [ "$(server_count)" -ne 0 ] && [ $waited -ne $MAX_SLEEP ] ; do
    ((waited+=1))
    sleep 1
  done
  [ "$(server_count)" -eq 0 ]
}

shutdown_servers() {
  local ports
  local dir
  local port
  ports="7000 7001 7002 7003 7004 7005 6379"
  if [ -d "$RUNTIME_ROOT" ]; then
    for dir in "$RUNTIME_ROOT"/*
    do
      [ -d "$dir" ] || continue
      port=${dir##*/}
      [[ "$port" =~ ^[0-9]+$ ]] && ports="$ports $port"
    done
  fi

  for i in $(printf '%s\n' $ports | sort -n -u)
  do
    redis-cli -p "$i" shutdown nosave >/dev/null 2>&1 || true
  done
}

prepare_runtime_root() {
  mkdir -p "$RUNTIME_ROOT" || {
    printf "${RED}Could not create runtime directory '$RUNTIME_ROOT'!${NC}\n"
    exit 8
  }
}

start_standalone_server() {
  local port=$1
  require_port_free "$port"

  printf "${BLUE}[Starting standalone server on port ${port}]${NC}\n"
  prepare_runtime_root
  cd "$RUNTIME_ROOT" || exit 8
  rm -rf "$port"
  mkdir -p "$RUNTIME_ROOT/$port"
  cd "$RUNTIME_ROOT/$port" || exit 8
  redis-server \
  --port "$port" \
  --protected-mode no \
  --cluster-enabled no \
  --save "" \
  --dbfilename "" \
  --appendonly no >/dev/null 2>&1 &

  # Wait until server is up
  if ! wait_for_server "$port"; then
    printf "${RED}Failed to bring up server on port ${port} - aborting!${NC}\n"
    exit 4
  fi
}

which redis-server >/dev/null
if [ $? -ne 0 ]; then
  printf "${RED}Could not locate redis-server binary!${NC}\n"
  exit 2
fi

case $1 in
  start|start_cluster)
    for i in {7000..7005}
    do
      require_port_free "$i"
    done
    if [ "$1" = "start" ]; then
      require_port_free "$SINGLE_PORT"
    fi
    
    printf "${BLUE}[Starting servers]${NC}\n"
    prepare_runtime_root
    cd "$RUNTIME_ROOT" || exit 8
    rm -rf {7000..7005}
    for i in {7000..7005}
    do
      mkdir -p "$RUNTIME_ROOT/$i"
      cd "$RUNTIME_ROOT/$i"
      redis-server \
      --port $i \
      --protected-mode no \
      --cluster-enabled yes \
      --cluster-config-file nodes.conf \
      --cluster-node-timeout 5000 \
      --save "" \
      --dbfilename "" \
      --appendonly no >/dev/null 2>&1 &

      # Wait until server is up
      if ! wait_for_server $i; then
        printf "${RED}Failed to bring up server on port $i - aborting!${NC}\n"
        exit 4
      fi
    done

    printf "${BLUE}[Starting cluster]${NC}\n"
    LOCAL_IP=$(hostname -I |cut -f1 -d" ")
    LOCAL_IP=${LOCAL_IP:-127.0.0.1}
    echo yes | redis-cli --cluster create ${LOCAL_IP}:7000 ${LOCAL_IP}:7001 ${LOCAL_IP}:7002 ${LOCAL_IP}:7003 ${LOCAL_IP}:7004 ${LOCAL_IP}:7005 --cluster-replicas 1 >/dev/null

    # Wait until cluster is up
    SLEEP=0
    while [ "$(redis-cli -p 7000 cluster nodes|wc -l)" -ne 6 ] && [ ${SLEEP} != ${MAX_SLEEP} ]; do
      ((SLEEP+=1))
      sleep 1
    done
    if [ $SLEEP == $MAX_SLEEP ]; then
      printf "${RED}Failed to create cluster - aborting!${NC}\n"
      exit 5
    fi

    SLEEP=0
    while [ "$(redis-cli -p 7000 cluster info|head -n1)" == "cluster_state:ok\r" ] && [ ${SLEEP} != ${MAX_SLEEP} ]; do
      ((SLEEP+=1))
      sleep 1
    done
    if [ $SLEEP == $MAX_SLEEP ]; then
      printf "${RED}Failed to bring up cluster - aborting!${NC}\n"
      exit 5
    fi

    if [ "$1" = "start" ]; then
      start_standalone_server "$SINGLE_PORT"
    fi

    printf "${GREEN}[Ready]${NC}\n"
    ;;
  start_single)
    # Single standalone redis-server, useful for non-cluster client tests.
    start_standalone_server "$SINGLE_PORT"
    printf "${GREEN}[Ready]${NC}\n"
    ;;
  stop)
    printf "${BLUE}[Stopping servers]${NC}\n"
    shutdown_servers
    if ! wait_for_servers_to_stop; then
      printf "${BLUE}[Graceful shutdown timed out; killing servers]${NC}\n"
      pkill -x redis-server
    fi

    if ! wait_for_servers_to_stop; then
      printf "${RED}Failed to kill all servers!${NC}\n"
      exit 6
    fi
    if [ -d "$RUNTIME_ROOT" ]; then
      for dir in "$RUNTIME_ROOT"/*
      do
        [ -d "$dir" ] || continue
        port=${dir##*/}
        [[ "$port" =~ ^[0-9]+$ ]] && rm -rf "$dir"
      done
    fi
    ;;
  *)
    printf "${RED}Unknown command ${1}${NC}\n"
    exit 7
    ;;
esac

exit 0

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

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [start | stop]"
  exit 1
fi

which redis-server >/dev/null
if [ $? -ne 0 ]; then
  printf "${RED}Could not locate redis-server binary!${NC}\n"
  exit 2
fi

case $1 in
  start)
    # Check if a redis server is already running
    if [ $(ps -aux|grep redis-server|wc -l) -ne 1 ]; then
      printf "${RED}Redis server already running - aborting!${NC}\n"
      exit 3
    fi
    
    printf "${BLUE}[Starting servers]${NC}\n"
    cd /tmp
    rm -rf {7000..7005}
    for i in {7000..7005}
    do
      mkdir -p /tmp/$i
      cd /tmp/$i
      redis-server \
      --port $i \
      --cluster-enabled yes \
      --cluster-config-file nodes.conf \
      --cluster-node-timeout 5000 \
      --save "" --appendonly no >/dev/null 2>&1 &

      # Wait until server is up
      SLEEP=0
      while [ "$(redis-cli -p $i ping 2>/dev/null)" != "PONG" ] && [ ${SLEEP} != ${MAX_SLEEP} ]; do
        ((SLEEP+=1))
        sleep 1
      done
      if [ $SLEEP == $MAX_SLEEP ]; then
        printf "${RED}Failed to bring up server on port $i - aborting!${NC}\n"
        exit 4
      fi
    done
    
    printf "${BLUE}[Starting cluster]${NC}\n"
    echo yes | redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 --cluster-replicas 1 >/dev/null

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

    printf "${GREEN}[Ready]${NC}\n"
    ;;
  stop)
    printf "${BLUE}[Stopping cluster]${NC}\n"
    pkill redis-server
    SLEEP=0
    while [ $(ps -aux|grep redis-server|wc -l) -ne 1 ] && [ $SLEEP -ne $MAX_SLEEP ] ; do
      ((SLEEP+=1))
      sleep 1
    done
    if [ $(ps -aux|grep redis-server|wc -l) -ne 1 ]; then
      printf "${RED}Failed to kill all servers!${NC}\n"
      exit 6
    fi
    ;;
  *)
    printf "${RED}Unknown command ${1}${NC}\n"
    exit 7
    ;;
esac

exit 0

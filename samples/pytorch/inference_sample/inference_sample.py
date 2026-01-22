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

import sys
import os.path
import torch
import nvtx
import argparse
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorch_samples_common as common
from queue import Queue
from enum import Enum

# Create a simple model with embedding layers
# remote interface is an object that implements the RemoteTable interface
# If no remote interface is provided, the embedding layer will be a uvm layer
# weight_init is an optional tensor that will be used to initialize the embedding layer, unless remote interface is provided in that case the weight_init will be ignored.
# cache_size is the size of the cache in bytes, if 0 then the embedding layer will be a gpu layer
class SimpleModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, *, data_type : torch.dtype = torch.float32, cache_size: int = 1024**3, remote_interface = None, weight_init : torch.Tensor = None):
        super(SimpleModel, self).__init__()
        if remote_interface is None:
            self.embedding = nve_layers.NVEmbedding(num_embeddings, embedding_dim, data_type, nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=weight_init, optimize_for_training=False)
        else:
            self.embedding = nve_layers.NVEmbedding(num_embeddings, embedding_dim, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=remote_interface, weight_init=weight_init, optimize_for_training=False)

    def forward(self, x):
        x = self.embedding(x)
        return x

class Job:
    class Type(Enum):
        Inference = 1
        Update = 2
    
    def __init__(self, bag : torch.Tensor, stream: torch.cuda.Stream, type: Type, job_index: int, *, update : torch.Tensor = None):
        self.bag = bag
        self.update = update
        self.stream = stream
        self.type = type
        self.job_index = job_index


class ConsumerThread:
    def __init__(self, queue: Queue, model: SimpleModel, stream: torch.cuda.Stream):
        self.queue = queue
        self.model = model
        self.stream = stream
        self.thread = threading.Thread(target=self.consumer)

    def consumer(self):
        with torch.cuda.stream(self.stream):
            while True:
                job : Job = self.queue.get()
                if job is None:
                    break
                if job.job_index % 100 == 0:
                    print(f"processing job {job.job_index}")
                if (job.type == Job.Type.Inference):
                    if job.stream != self.stream:
                        torch.cuda.current_stream().wait_stream(job.stream)
                    x = self.model(job.bag)
                elif (job.type == Job.Type.Update):
                    self.model.embedding.update(job.bag, job.update)
        
    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()
        self.stream.synchronize()

# class that produces jobs for the consumer thread
# a job is can be either an inference job or an update job
# an inference job is a job that does an inference on a bag of embeddings, the producer will generate a power law distribution of key
# an update job is a job that updates a number of embeddings, the producer will generate of uniformly distributed keys every random frequency iterations
class ProducerThread:
    def __init__(self, queue: Queue, num_embeddings: int, embedding_dim: int, alpha: float, hotness: int, batch_size: int, num_iterations: int, data_type: torch.dtype, stream: torch.cuda.Stream, thread_index: int, update_frequency: int, update_size: int):
        self.queue = queue
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.hotness = hotness
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.stream = stream
        self.data_type = data_type
        self.thread = threading.Thread(target=self.producer)
        self.thread_index = thread_index
        self.update_frequency = update_frequency
        self.update_size = update_size

    def producer(self):
        with torch.cuda.stream(self.stream):
            for i in range(self.num_iterations):
                job_index = self.thread_index * self.num_iterations + i
                if job_index % 100 == 0:
                    print(f"queueing inference job {job_index}")
                if self.update_frequency != 0 and job_index % self.update_frequency == 0 and job_index != 0:
                    # send update job
                    print(f"queueing update job {job_index}")
                    bag = torch.randint(1, self.num_embeddings, (self.update_size,), device="cuda", dtype=torch.int64)
                    updates = torch.randn(self.update_size, self.embedding_dim, device="cuda", dtype=self.data_type)
                    self.queue.put(Job(bag, torch.cuda.current_stream(), Job.Type.Update, job_index, update=updates))
                else:
                    # send inference job
                    bag = common.PowerLaw(1, self.num_embeddings, self.alpha, self.batch_size * self.hotness)
                    self.queue.put(Job(bag, torch.cuda.current_stream(), Job.Type.Inference, job_index))
            #passing none as a job to signal the consumer to exit
            self.queue.put(None)

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()
        self.stream.synchronize() 

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-ps", type=str, default="", help="Path to the PS")
    parser.add_argument("--embedding-dim", "-dim", type=int, default=128, help="Dimension of the embedding in elements")
    parser.add_argument("--num-embeddings", "-n", type=int, default=1000000, help="Number of embeddings")
    parser.add_argument("--cache-size", "-c", type=int, default=1024**3, help="Size of the cache in bytes")
    parser.add_argument("--data-type", "-dt", type=str, default="float32", help="Data type of the embedding", choices=["float32", "float16"])
    parser.add_argument("--num-threads", "-t", type=int, default=1, help="Number of threads")
    parser.add_argument("--alpha", type=float, default=1.02, help="Alpha for the power law distribution")
    parser.add_argument("--hotness", type=int, default=1024, help="Hotness of the embedding")
    parser.add_argument("--batch-size", "-bs", type=int, default=16, help="Batch size")
    parser.add_argument("--num-iterations", "-ni", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--use-ps", "-ps", action="store_true", default=False, help="Use remote interface")
    parser.add_argument("--update-frequency", "-uf", type=int, default=0, help="Update frequency, every x iterations a number of embeddings will be updated, a frequency of 0 means no updates")
    parser.add_argument("--update-size", "-us", type=int, default=16, help="Number of embeddings to update")
    args = parser.parse_args(args)

    if args.use_ps:
        remote_interface = common.get_remote_interface(args.path_to_ps, args.num_embeddings, args.embedding_dim, common.data_type_to_torch_dtype(args.data_type))
    else:
        remote_interface = None

    model = SimpleModel(args.embedding_dim, args.num_embeddings, data_type=common.data_type_to_torch_dtype(args.data_type), cache_size=args.cache_size, remote_interface=remote_interface)

    num_iterations_per_thread = args.num_iterations // args.num_threads
    consumer_threads = []
    producer_threads = []
    start_time = time.perf_counter()
    for i in range(args.num_threads):
        queue = Queue()
        stream = torch.cuda.Stream()
        producer_thread = ProducerThread(queue, args.num_embeddings, args.embedding_dim, args.alpha, args.hotness, args.batch_size, num_iterations_per_thread, common.data_type_to_torch_dtype(args.data_type), stream, i, args.update_frequency, args.update_size)
        consumer_thread = ConsumerThread(queue, model, stream)
        consumer_threads.append(consumer_thread)
        producer_threads.append(producer_thread)

        producer_thread.start()
        consumer_thread.start()
    
    for consumer_thread in consumer_threads:
        consumer_thread.join()

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds.")
    print(f"QPS: {(args.hotness * args.batch_size * args.num_iterations) / (end_time - start_time) / 1e6} Mega QPS")

    for producer_thread in producer_threads:
        producer_thread.join()
    print("done")

if __name__ == "__main__":
    main(sys.argv[1:])


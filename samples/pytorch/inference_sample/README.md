# Inference Sample

The aim of this sample is to provide an example of performing inference using NV Embedding Cache Python package. 
The sample performs multi-thread serving based on random, power law distributed, input generation of a simple embedding model.
The sample also shows how to perform online asynchronous weights updates.

## Installation
    ``` pip install . ```

## Usage Example
```console
    inference_sample.py --batch-size 1024 --hotness 32 --num-iterations 1000 --cache-size 8192 --data-type float32
```
Output:

```console
    Time taken: 2.8852698470000178 seconds.
    QPS: 11.35699665460088 Mega QPS
    done
```

Notes:
1. Results may vary based on your system
2. Results also include time to generate inputs

## Using Parameter Server
Use the --use-ps option to enable parameter server in the sample, first the supplied parameter server bindings need to be compiled, see [mock_ps_binding.cpp](../mock_ps_binding.cpp) (they are automatically built if the entire project is created).
Use --path-to-ps (e.g ./build if used "pip install ." from main folder) to supply a specific path to the generated .so or, if not supplied, let the sample search for one.
The supplied mock implementation is a simple single-threaded system memory example of how to interface with a parameter server. Users can implement their own; see [python_api.md](../../../docs/python_api.md) for more information.


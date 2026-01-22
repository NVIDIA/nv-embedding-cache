# Import embedding saved by Dynamic Embedding (C++)

This sample imports an embedding saved by training with Dynamic Embedding (published in [https://github.com/NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples)).
The keys and values are read from a pair of NumPy files.
You can also generate random keys/values with the [gen_np_files.py](gen_np_files.py) script.

The sample will create a hierarchical embedding layer with a small GPU cache and host table. The imported values will be used to initialize the host table (with the insert() API). Following the initialization, the sample will verify all imported keys are available in the layer and return the correct values when calling lookup().

To run the sample call:

```bash
$ ./build/bin/import_sample -kf /tmp/keys.npy -vf /tmp/values.npy --verbose
Importing files (/tmp/keys.npy, /tmp/values.npy)
Found 1048576 keys
Creating GPU table
Creating Host table
Creating Hierarchical embedding layer
Importing keys/values
Checking import
Done!
```

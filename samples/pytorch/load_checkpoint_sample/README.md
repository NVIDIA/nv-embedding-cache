# Load Checkpoint Sample
This sample shows how to load a dcp checkpoint of a saved embedding table to NVE Embedding Layer.
The sample uses a ToyModel which is composed of two components: a sparse part (including an Embedding Table) and a dense part.
The sparse part of the ToyModel has three implementations: 1. Torch.nn.Embedding, 2. TorchRec.EmbeddingCollection, 3. NVEmbedding. The first two implementations are used for training, while NVE is used for inference.
Training mode can be controlled via the switch --use_torchrec

The sample has two modes Save/Load:
1. In save mode the sample takes the ToyModel and saves its trained weights to disk using torch.distributed.checkpoint (dcp)
2. In load mode the sample loads the dcp into an NVE implementation of the ToyModel 

## Usage Example
saving: 
``` ./load_sample.py --mode save --use_torchrec ```
loading: 
``` ./load_sample.py --mode load --use_torchrec ```

Note: it is important to use the same "training mode" for both save and load as it represents the saved model on disk.



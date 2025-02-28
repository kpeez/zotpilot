import torch

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_MAX_TOKENS = 512
BATCH_SIZE = 32
DEVICE = "auto"


def get_device():
    if DEVICE == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return DEVICE

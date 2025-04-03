import torch

# Text extraction and embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_MAX_TOKENS = 512
BATCH_SIZE = 32
DEVICE = "auto"
# LLM
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1000


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

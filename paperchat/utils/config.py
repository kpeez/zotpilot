import torch

# Text extraction and embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_MAX_TOKENS = 512
BATCH_SIZE = 32
DEVICE = "auto"

# LLM
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# System prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful academic research assistant helping with scientific papers.
Act as an expert in the field of the paper you're discussing.
Answer questions based on your knowledge of the paper and the context provided.
When citing information, refer to the specific section of the paper where it appears.
Be precise, scholarly, and focus on the factual content of the paper.
"""


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

from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np
import torch

def get_device() -> str:
    """
    Automatically detect and return the best available device (CPU or GPU).
    
    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class GermanEmbeddings:
    def __init__(self, device: str = None):
        """
        Initialize the German embeddings model using multilingual MiniLM.
        
        Args:
            device (str, optional): Device to use ('cuda' or 'cpu'). 
                                  If None, automatically detects the best available device.
        """
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.device = device if device is not None else get_device()
        
        # Print device information
        if self.device == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for embeddings")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: The dimension of the embedding vectors
        """
        # For MiniLM-L12-v2, the dimension is 384
        return 384

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Convert to numpy arrays for computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity) 
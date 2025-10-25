 # liuembeddings/embeddings.py
import tensorflow_hub as hub
import numpy as np

class LiuEmbeddings:
    """
    TensorFlow-based embeddings using Universal Sentence Encoder (USE)
    """

    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        self.model = hub.load(model_url)
        print(f"✅ Loaded TF model from {model_url}")

    def embed_query(self, text: str):
        """
        Embed a single query string
        """
        vec = self.model([text]).numpy()[0]
        return vec.tolist()

    def embed_documents(self, texts):
        """
        Embed a list of text documents/chunks
        """
        vectors = self.model(texts).numpy()
        return [v.tolist() for v in vectors]

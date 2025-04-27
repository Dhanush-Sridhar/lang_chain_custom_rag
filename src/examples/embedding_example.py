from src.utils.embeddings import GermanEmbeddings
import time

def benchmark_embeddings(embeddings, texts, device_name):
    """Benchmark embedding generation performance."""
    print(f"\nBenchmarking on {device_name}...")
    
    # Benchmark single text embedding
    start_time = time.time()
    single_embedding = embeddings.embed_text(texts[0])
    single_time = time.time() - start_time
    print(f"Single text embedding time: {single_time:.4f} seconds")
    
    # Benchmark multiple document embeddings
    start_time = time.time()
    document_embeddings = embeddings.embed_documents(texts)
    batch_time = time.time() - start_time
    print(f"Batch embedding time: {batch_time:.4f} seconds")
    
    return single_time, batch_time

def main():
    # Example German texts
    german_texts = [
        "Das ist ein Beispieltext auf Deutsch.",
        "Dies ist ein weiterer deutscher Text.",
        "Die Sonne scheint heute sehr hell.",
        "Der Computer ist ein nützliches Werkzeug.",
        "Die künstliche Intelligenz entwickelt sich schnell.",
        "Maschinelles Lernen ist ein spannendes Gebiet.",
        "Neuronale Netze können komplexe Muster erkennen.",
        "Deep Learning revolutioniert viele Bereiche.",
        "Die Datenverarbeitung wird immer effizienter.",
        "Algorithmen lernen aus großen Datenmengen."
    ]
    
    # Initialize embeddings with automatic device detection
    embeddings = GermanEmbeddings()
    
    # Generate embeddings and compute similarity
    single_embedding = embeddings.embed_text(german_texts[0])
    print(f"Single text embedding dimension: {len(single_embedding)}")
    
    document_embeddings = embeddings.embed_documents(german_texts)
    print(f"Number of document embeddings: {len(document_embeddings)}")
    
    similarity = embeddings.compute_similarity(
        document_embeddings[0],
        document_embeddings[1]
    )
    print(f"Similarity between first two texts: {similarity:.4f}")
    
    # Benchmark performance
    benchmark_embeddings(embeddings, german_texts, "GPU" if embeddings.device == 'cuda' else "CPU")

if __name__ == "__main__":
    main() 
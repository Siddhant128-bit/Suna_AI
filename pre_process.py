import sounddevice as sd
from scipy.io.wavfile import write
import os
from . import embedding_extractor
import numpy as np 
import pickle
import faiss



def record_and_dump(name='test',duration=5,fs=44100,count=0):
    begin=input(f'Press Enter to begin {count+1 } voice!')
    myrecording=sd.rec(int(duration*fs),samplerate=fs,channels=2)
    sd.wait()
    write(f'{name}.wav',fs,myrecording)


def create_vectored_database(database_path='Database'):
    # Initialize FAISS index (L2 normalized embeddings for cosine similarity)
    embedding_dim = 192  # Dimension of ECAPA-TDNN embeddings
    index = faiss.IndexFlatIP(embedding_dim)  # IP = Inner Product (cosine similarity with normalized embeddings)

    # Function to add embeddings to FAISS
    def add_to_index(embedding, speaker_id, index, metadata):
        """
        Add an embedding to the FAISS index and store metadata.
        """
        embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
        index.add(np.array([embedding], dtype=np.float32))  # Add to FAISS
        metadata.append(speaker_id)  # Save speaker ID for reference

    # Metadata storage
    metadata = []
    # Example usage (Add 5 embeddings from authenticated speaker)
    for file in os.listdir(database_path):
        emb = embedding_extractor.extract_embedding(f'Database/{file}')
        add_to_index(emb, speaker_id="authenticated_user", index=index, metadata=metadata)

    print(f"Index contains {index.ntotal} embeddings.")
    return index,metadata

def save_vectored_database(index, metadata, index_file='faiss_index.bin', metadata_file='metadata.pkl'):
    """
    Save the FAISS index and metadata to files.
    """
    # Save FAISS index
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}.")

    # Save metadata using pickle
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_file}.")


def load_vectored_database(index_file='faiss_index.bin', metadata_file='metadata.pkl'):
    """
    Load the FAISS index and metadata from files.
    """
    # Load FAISS index
    index = faiss.read_index(index_file)
    print(f"FAISS index loaded from {index_file}.")

    # Load metadata using pickle
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Metadata loaded from {metadata_file}.")
    
    return index, metadata



def storing_audio_db(total_samples=5):
    try:
        os.mkdir('Database')
    except:
        pass

    for i in range(0,total_samples):
        record_and_dump(name=f'Database/sample_{i}',count=i)

    vectored_db,metadata=create_vectored_database()
    save_vectored_database(index=vectored_db,metadata=metadata)



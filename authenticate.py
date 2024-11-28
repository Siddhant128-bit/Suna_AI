from . import pre_process
from . import embedding_extractor
import numpy as np 



def verify_speaker(test_audio_path='test.wav',number_of_samples=3, threshold=0.8):
    """
    Verify if a test speaker matches the authenticated speaker.
    """
    index,metadata=pre_process.load_vectored_database()
    # Extract embedding for the test audio
    test_embedding = embedding_extractor.extract_embedding(test_audio_path)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
    
    # Search the FAISS index
    distances, indices = index.search(np.array([test_embedding], dtype=np.float32), k=number_of_samples)  # Top 3 matches
    mean_similarity = np.mean(distances)  # Average similarity
    
    print("Cosine Similarities:", distances.flatten())
    print('mean_similarity:',mean_similarity)
    print('threshold: ',threshold)
    if mean_similarity > threshold:
        speaker_id = metadata[indices[0][0]]
        return 1
    else:
        return 0

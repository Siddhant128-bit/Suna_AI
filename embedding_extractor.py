from speechbrain.pretrained import SpeakerRecognition
import numpy as np
import librosa
import torch

# Loading pretrained model.
verifier = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

def extract_embedding(audio_path):
    """
    Extracting embeddings using pretrained speechbrain models.
    """
    signal, sample_rate = librosa.load(audio_path, sr=16000)  
    signal = torch.tensor(signal).unsqueeze(0)
    embedding = verifier.encode_batch(signal)
    return embedding.squeeze().numpy()  

import torchaudio

def wav_loads(path):
    from glob import glob
    return list(map(torchaudio.load(path), glob(path+'*.wav')))
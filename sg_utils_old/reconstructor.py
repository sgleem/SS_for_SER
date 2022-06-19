import librosa
from tqdm import tqdm

class MelSpec2Wav:
    """
    Reconstructed by Griffin-Lim algorithm
    """
    def __init__(self, *args, **kwargs):
        self.sr = kwargs.get("sampling_rate", 16000)
        self.n_fft = kwargs.get("window_size", 400) # 16 * 25ms
        self.hop_length = kwargs.get("step_size", 160) # 16 * 10ms
        return

    def reconstruct_all(self, mel_spec_list):
        raw_wav_list = []
        for mel_spec in mel_spec_list:
            raw_wav = librosa.feature.inverse.mel_to_audio(M=mel_spec, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            raw_wav_list.append(raw_wav)
        result = raw_wav_list
        return result
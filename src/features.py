import torch
import torchaudio


def _to_waveform_tensor(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 3:
        wav = wav.mean(dim=1)

    return wav


def _get_window_fn(window: str):
    if window == "hamming":
        return torch.hamming_window

    return torch.hann_window


def log_mel(
    batch, # (B, wav_max_length)
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    window: str = "hann",
):
    waveform = _to_waveform_tensor(batch)
    window_fn = _get_window_fn(window)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        window_fn=window_fn,
    ).to(waveform.device)

    mel_spec = mel_transform(waveform) # (B, n_mels, T)

    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(waveform.device)
    log_mel_spec = amplitude_to_db(mel_spec)

    return log_mel_spec.transpose(-1, -2)


def mfcc(
    batch,  # (B, wav_max_length)
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    window: str = "hann",
):
    waveform = _to_waveform_tensor(batch)
    window_fn = _get_window_fn(window)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "window_fn": window_fn,
        },
    ).to(waveform.device)

    mfcc_features = mfcc_transform(waveform)

    return mfcc_features.transpose(-1, -2)


FEATURE_FN = {
    "log_mel": log_mel,
    "mfcc": mfcc,
}


def get_features(batch, cfg):
    sr = cfg.dataset.sample_rate

    outputs = []

    for extractor_cfg in cfg.feature.extractors:
        feature_fn = FEATURE_FN[extractor_cfg.name]
        params = extractor_cfg.get("params", {})
        outputs.append(feature_fn(batch, sr, **params))

    return torch.cat(outputs, dim=-1)

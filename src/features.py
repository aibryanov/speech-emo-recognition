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


def _get_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    power: float = 2.0,
) -> torch.Tensor:
    window_fn = _get_window_fn(window)
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        window_fn=window_fn,
    ).to(waveform.device)

    return spectrogram(waveform)


def _get_frequency_bins(num_bins: int, sr: int, device, dtype) -> torch.Tensor:
    return torch.linspace(0, sr / 2, steps=num_bins, device=device, dtype=dtype)


def _get_magnitude_spectrogram(
    batch,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> torch.Tensor:
    waveform = _to_waveform_tensor(batch)

    return _get_spectrogram(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=1.0,
    )


def _get_power_spectrogram(
    batch,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
) -> torch.Tensor:
    waveform = _to_waveform_tensor(batch)

    return _get_spectrogram(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=2.0,
    )


def apply_feature_regularization(features: torch.Tensor, cfg, training: bool = False) -> torch.Tensor:
    regularization_cfg = cfg.feature.get("regularization")
    if regularization_cfg is None or not regularization_cfg.get("enabled", False) or not training:
        return features

    augmented = features
    noise_std = regularization_cfg.get("feature_noise_std", 0.0)
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std

    augmented = augmented.transpose(1, 2)

    time_mask_param = regularization_cfg.get("time_mask_param", 0)
    for _ in range(regularization_cfg.get("num_time_masks", 0)):
        if time_mask_param > 0:
            augmented = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)(augmented)

    freq_mask_param = regularization_cfg.get("freq_mask_param", 0)
    for _ in range(regularization_cfg.get("num_freq_masks", 0)):
        if freq_mask_param > 0:
            augmented = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)(augmented)

    augmented = augmented.transpose(1, 2)

    feature_dropout = regularization_cfg.get("feature_dropout", 0.0)
    if feature_dropout > 0:
        augmented = torch.nn.functional.dropout(augmented, p=feature_dropout, training=True)

    return augmented


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


def mfcc_delta(
    batch,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    window: str = "hann",
):
    base_mfcc = mfcc(
        batch,
        sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window=window,
    )

    return torchaudio.functional.compute_deltas(base_mfcc.transpose(-1, -2)).transpose(-1, -2)


def mfcc_delta2(
    batch,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    window: str = "hann",
):
    delta = mfcc_delta(
        batch,
        sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window=window,
    )

    return torchaudio.functional.compute_deltas(delta.transpose(-1, -2)).transpose(-1, -2)


def rms_energy(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
):
    del sr
    power_spec = _get_power_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window)
    rms = power_spec.mean(dim=1).clamp_min(1e-8).sqrt()

    return torch.log1p(rms).unsqueeze(-1)


def spectral_centroid(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
):
    magnitude_spec = _get_magnitude_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window)
    freqs = _get_frequency_bins(
        magnitude_spec.shape[1],
        sr,
        magnitude_spec.device,
        magnitude_spec.dtype,
    ).view(1, -1, 1)
    centroid_hz = (magnitude_spec * freqs).sum(dim=1) / magnitude_spec.sum(dim=1).clamp_min(1e-8)

    return (centroid_hz / (sr / 2)).unsqueeze(-1)


def spectral_bandwidth(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
):
    magnitude_spec = _get_magnitude_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window)
    freqs = _get_frequency_bins(
        magnitude_spec.shape[1],
        sr,
        magnitude_spec.device,
        magnitude_spec.dtype,
    ).view(1, -1, 1)
    centroid_hz = (magnitude_spec * freqs).sum(dim=1, keepdim=True) / magnitude_spec.sum(dim=1, keepdim=True).clamp_min(1e-8)
    bandwidth_hz = (
        (magnitude_spec * (freqs - centroid_hz).pow(2)).sum(dim=1)
        / magnitude_spec.sum(dim=1).clamp_min(1e-8)
    ).clamp_min(1e-8).sqrt()

    return (bandwidth_hz / (sr / 2)).unsqueeze(-1)


def spectral_rolloff(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
    window: str = "hann",
):
    magnitude_spec = _get_magnitude_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window)
    freqs = _get_frequency_bins(
        magnitude_spec.shape[1],
        sr,
        magnitude_spec.device,
        magnitude_spec.dtype,
    )
    cumulative_energy = magnitude_spec.cumsum(dim=1)
    threshold = magnitude_spec.sum(dim=1, keepdim=True) * roll_percent
    rolloff_indices = (cumulative_energy >= threshold).float().argmax(dim=1)
    rolloff_hz = freqs[rolloff_indices]

    return (rolloff_hz / (sr / 2)).unsqueeze(-1)


def spectral_flatness(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
):
    del sr
    magnitude_spec = _get_magnitude_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window).clamp_min(1e-8)
    geometric_mean = torch.exp(torch.log(magnitude_spec).mean(dim=1))
    arithmetic_mean = magnitude_spec.mean(dim=1).clamp_min(1e-8)
    flatness = geometric_mean / arithmetic_mean

    return flatness.unsqueeze(-1)


def spectral_flux(
    batch,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
):
    del sr
    magnitude_spec = _get_magnitude_spectrogram(batch, n_fft=n_fft, hop_length=hop_length, window=window)
    normalized_spec = magnitude_spec / magnitude_spec.sum(dim=1, keepdim=True).clamp_min(1e-8)
    previous_frame = torch.nn.functional.pad(normalized_spec[:, :, :-1], (1, 0))
    flux = (normalized_spec - previous_frame).pow(2).sum(dim=1).sqrt()

    return torch.log1p(flux).unsqueeze(-1)


FEATURE_DIMS = {
    "log_mel": lambda params: params.get("n_mels", 80),
    "mfcc": lambda params: params.get("n_mfcc", 40),
    "mfcc_delta": lambda params: params.get("n_mfcc", 40),
    "mfcc_delta2": lambda params: params.get("n_mfcc", 40),
    "rms_energy": lambda params: 1,
    "spectral_centroid": lambda params: 1,
    "spectral_bandwidth": lambda params: 1,
    "spectral_rolloff": lambda params: 1,
    "spectral_flatness": lambda params: 1,
    "spectral_flux": lambda params: 1,
}


def get_feature_dim(extractor_cfg) -> int:
    if extractor_cfg.name not in FEATURE_DIMS:
        raise ValueError(f"Unsupported feature extractor: {extractor_cfg.name}")

    return FEATURE_DIMS[extractor_cfg.name](extractor_cfg.get("params", {}))


FEATURE_FN = {
    "log_mel": log_mel,
    "mfcc": mfcc,
    "mfcc_delta": mfcc_delta,
    "mfcc_delta2": mfcc_delta2,
    "rms_energy": rms_energy,
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    "spectral_rolloff": spectral_rolloff,
    "spectral_flatness": spectral_flatness,
    "spectral_flux": spectral_flux,
}


def get_features(batch, cfg, training: bool = False):
    sr = cfg.dataset.sample_rate

    outputs = []

    for extractor_cfg in cfg.feature.extractors:
        feature_fn = FEATURE_FN[extractor_cfg.name]
        params = extractor_cfg.get("params", {})
        outputs.append(feature_fn(batch, sr, **params))

    features = torch.cat(outputs, dim=-1)

    return apply_feature_regularization(features, cfg, training=training)

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", palette="bright")


def _to_waveform_tensor(wav) -> torch.Tensor:
    wav = torch.as_tensor(wav, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    return wav


def _to_plot_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32).detach().cpu()


def _label_text(label) -> str:
    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            return str(label.item())
        return str(label.detach().cpu().tolist())
    return str(label)


def show_waveform(wav, label, sr: int = 22050) -> None:
    wav = _to_waveform_tensor(wav).detach().cpu()
    time_axis = torch.arange(wav.numel(), dtype=torch.float32) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis.numpy(), wav.numpy(), linewidth=1.0)
    plt.ylabel("Амплитуда")
    plt.xlabel("Время (секунды)")
    plt.title(_label_text(label))
    plt.tight_layout()
    plt.show()


def show_spectrogram(
    wav,
    label,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> None:
    wav = _to_waveform_tensor(wav)

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )(wav)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB(stype="power")(spectrogram)
    spectrogram_db = spectrogram_db.detach().cpu()

    plt.figure(figsize=(12, 5))
    plt.imshow(
        spectrogram_db.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, wav.numel() / sr, 0, sr / 2],
    )
    plt.title(_label_text(label))
    plt.ylabel("Частота (Гц)")
    plt.xlabel("Время (секунды)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels) -> None:
    plt.figure(figsize=(10, 3))
    plt.title("Распределение классов")
    sns.countplot(x=labels)
    plt.show()


def plot_duration_distribution(durations, bins: int = 30, kde: bool = True) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=bins, kde=kde)
    plt.title("Распределение длительностей аудио в секундах")
    plt.xlabel("Длительность (секунды)")
    plt.ylabel("Количество")
    plt.grid(alpha=0.3)
    plt.show()


def plot_mel_filter_bank(mel_fb, fmin: float, fmax: float) -> None:
    mel_fb = _to_plot_tensor(mel_fb)

    plt.figure(figsize=(12, 5))
    plt.imshow(mel_fb.numpy(), aspect="auto", origin="lower")
    plt.colorbar(label="weight")
    plt.xlabel("FFT bin")
    plt.ylabel("Mel filter index")
    plt.title("Mel filter bank")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.imshow(
        mel_fb.numpy(),
        aspect="auto",
        origin="lower",
        extent=[fmin, fmax, 0, mel_fb.shape[0]],
    )
    plt.colorbar(label="weight")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mel filter index")
    plt.title("Mel filter bank (x-axis in Hz)")
    plt.tight_layout()
    plt.show()


def plot_log_mel_spectrogram(log_mel_spec, sr: int, num_samples: int) -> None:
    log_mel_spec = _to_plot_tensor(log_mel_spec)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        log_mel_spec.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, num_samples / sr, 0, log_mel_spec.shape[0]],
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Время (секунды)")
    plt.ylabel("Mel bin")
    plt.tight_layout()
    plt.show()


def plot_spectrogram_db(spec_db, sr: int, num_samples: int) -> None:
    spec_db = _to_plot_tensor(spec_db)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spec_db.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, num_samples / sr, 0, sr / 2],
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Время (секунды)")
    plt.ylabel("Частота (Гц)")
    plt.tight_layout()
    plt.show()

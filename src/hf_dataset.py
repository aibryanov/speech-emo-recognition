"""
Загружаем датасеты с HF
"""

from datasets import load_dataset


hf_audioMNIST = load_dataset("gilkeyio/AudioMNIST")
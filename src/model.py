import torch.nn as nn
from src.features import get_feature_dim, get_features


def _get_feature_dim(cfg):
    return sum(get_feature_dim(extractor_cfg) for extractor_cfg in cfg.feature.extractors)


def _get_lstm_output_dim(cfg):
    num_directions = 2 if cfg.model.lstm.bidirectional else 1

    return cfg.model.lstm.hidden_size * num_directions


def _get_num_classes(cfg):
    return cfg.dataset.get("num_classes", cfg.model.num_classes)


class Featurizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dropout = nn.Dropout(self.cfg.model.get("feature_dropout", 0.0))

    def forward(self, batch):
        features = get_features(batch, self.cfg, training=self.training)

        return self.input_dropout(features)


class LSTMClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.features = Featurizer(self.cfg)
        lstm_cfg = self.cfg.model.lstm
        dropout = lstm_cfg.dropout if lstm_cfg.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=_get_feature_dim(self.cfg),
            hidden_size=lstm_cfg.hidden_size,
            num_layers=lstm_cfg.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=lstm_cfg.bidirectional,
        )
        self.pooling = self.cfg.model.pooling
        self.head_dropout = nn.Dropout(self.cfg.model.head_dropout)
        self.fc = nn.Linear(_get_lstm_output_dim(self.cfg), _get_num_classes(self.cfg))

    def _pool_outputs(self, outputs):
        if self.pooling == "last":
            return outputs[:, -1, :]
        if self.pooling == "mean":
            return outputs.mean(dim=1)
        if self.pooling == "max":
            return outputs.max(dim=1).values

        raise ValueError(f"Unsupported pooling type: {self.pooling}")

    def forward(self, wavs):
        features = self.features(wavs)
        outputs, _ = self.lstm(features)
        pooled = self._pool_outputs(outputs)
        logits = self.fc(self.head_dropout(pooled))

        return logits

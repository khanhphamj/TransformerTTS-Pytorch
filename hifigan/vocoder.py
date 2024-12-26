# hifigan/vocoder.py

import json
import torch
from hifigan.models import Generator

class AttrDict(dict):
    """
    Để truy cập h.key thay vì h['key']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class HiFiGANVocoder:
    def __init__(self, checkpoint_path, config_path, device='cpu'):
        """
        Load HiFi-GAN Generator với config V1 chuẩn (JSON).
        Checkpoint thường có key "generator".
        """
        self.device = device

        # 1) Đọc file config (JSON) -> AttrDict
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        h = AttrDict(config_dict)

        # 2) Tạo model Generator
        self.model = Generator(h).to(device)

        # 3) Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        # checkpoint có dict_keys(['generator', 'discriminator', ...]) => load ckpt["generator"]
        self.model.load_state_dict(ckpt["generator"], strict=True)

        self.model.eval()
        if hasattr(self.model, "remove_weight_norm"):
            self.model.remove_weight_norm()

    @torch.no_grad()
    def infer(self, mel):
        """
        mel: torch.Tensor [1, n_mels, T] (đặt lên self.device)
        Trả về audio: torch.Tensor [samples]
        """
        mel = mel.to(self.device)
        audio = self.model(mel).squeeze(1)  # => [1, T]
        audio = audio[0]                    # => [T]
        return audio
import json
import torch
import soundfile as sf

from tts_transformer.model import TransformerTTS
from tts_transformer.dataset import character_set, char2idx
from hifigan.vocoder import HiFiGANVocoder


def text_to_sequence(text):
    seq = [char2idx.get(c, char2idx['<unk>']) for c in text]
    return torch.tensor([seq], dtype=torch.long)


def load_transformer_tts(tts_ckpt_path, tts_config_path, device='cpu'):
    with open(tts_config_path, "r") as f:
        cfg = json.load(f)
    # Tạo model TransformerTTS
    model = TransformerTTS(
        input_dim=len(character_set),
        output_dim=cfg["n_mels"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"]
    )
    state = torch.load(tts_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load config HiFi-GAN
    with open("config/hifigan.json", "r") as f:
        hifigan_cfg = json.load(f)

    # 2) Load vocoder
    vocoder = HiFiGANVocoder("checkpoints/hifigan_gen_universal.pth", hifigan_cfg)
    vocoder.model.to(device)

    # 3) Lấy Transformer TTS
    tts_model = load_transformer_tts(
        tts_ckpt_path="checkpoints/checkpoints_23_12_2024_TTS_Transformer/transformer_tts_epoch24.pth",
        tts_config_path="config/tts_transformer.json",
        device=device
    )

    # 4) Từ text -> mel
    text = "Xin chào các bạn!"
    text_tensor = text_to_sequence(text).to(device)

    # (Demo) Tạo dummy mel 1 bước
    n_mels = tts_model.fc_out.out_features
    dummy_mel_input = torch.zeros((1, n_mels, 1), device=device)
    with torch.no_grad():
        mel_out = tts_model(text_tensor, dummy_mel_input)  # [1, n_mels, T]

    # 5) Vocoder -> audio
    with torch.no_grad():
        audio_out = vocoder.infer(mel_out)  # [samples]

    # 6) Lưu WAV
    sf.write("output.wav", audio_out.cpu().numpy(), hifigan_cfg["sample_rate"])
    print("Đã lưu output.wav!")

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tts_transformer.dataset import TTSDataLoader, compute_mel_stats, collate_fn, character_set
from tts_transformer.model import TransformerTTS

def train(config_path="../config/config_tts.json"):
    # Đọc config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_name = cfg["dataset_name"]
    cache_dir = cfg["cache_dir"]
    split = cfg["split"]

    sample_rate = cfg["sample_rate"]
    n_mels = cfg["n_mels"]

    d_model = cfg["d_model"]
    nhead = cfg["nhead"]
    num_layers = cfg["num_layers"]
    dim_feedforward = cfg["dim_feedforward"]
    dropout = cfg["dropout"]

    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg["num_epochs"]
    grad_clip = cfg["grad_clip"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(character_set)
    output_dim = n_mels

    # Tính mean, std trên mel
    mel_mean, mel_std = compute_mel_stats(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        n_mels=output_dim,
        sample_rate=sample_rate
    )

    # Khởi tạo model
    model = TransformerTTS(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset & DataLoader
    train_dataset = TTSDataLoader(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        mel_mean=mel_mean,
        mel_std=mel_std,
        n_mels=output_dim,
        sample_rate=sample_rate
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)

    # Bắt đầu train
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (text_batch, mel_batch, text_lens, mel_lens) in enumerate(train_loader):
            text_batch, mel_batch = text_batch.to(device), mel_batch.to(device)

            optimizer.zero_grad()
            output = model(text_batch, mel_batch)
            loss = criterion(output, mel_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"[Epoch {epoch}] Step {i}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"=> Epoch {epoch} done! Avg Loss: {avg_loss:.4f}")

        # Lưu checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/transformer_tts_epoch{epoch}.pth")

if __name__ == "__main__":
    # Mặc định load ../config/config_tts.json
    train()

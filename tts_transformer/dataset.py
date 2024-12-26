import math
import torch
import time
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

################################################################
# Character set & mapping
################################################################
character_set = list(" aăâbcdđeêfghijklmnoôơpqrstuưvwxyzAĂÂBCDĐEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?")
if '<unk>' not in character_set:
    character_set.insert(0, '<unk>')
char2idx = {c: i for i, c in enumerate(character_set)}

################################################################
# Tính mean, std cho mel-spectrogram
################################################################
def compute_mel_stats(dataset_name, split, cache_dir="./dataset_cache", n_mels=80, sample_rate=16000):
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    sums = 0.0
    squared_sums = 0.0
    count = 0

    print(f"Computing mel stats (mean/std) over the dataset '{dataset_name}' (split: {split})...")
    for item in dataset:
        audio_array = item["audio"]["array"]
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        mel_spec = mel_transform(audio_tensor)
        sums += mel_spec.sum().item()
        squared_sums += (mel_spec ** 2).sum().item()
        count += mel_spec.numel()

    mean = sums / count
    var = (squared_sums / count) - (mean ** 2)
    std = math.sqrt(var)

    print(f"=> Computed mel stats: mean={mean:.4f}, std={std:.4f}")
    return mean, std

################################################################
# Dataset TTS
################################################################
class TTSDataLoader(Dataset):
    def __init__(
        self,
        dataset_name,
        split,
        cache_dir,
        mel_mean=None,
        mel_std=None,
        n_mels=80,
        sample_rate=16000,
        max_retries=5,
        retry_delay=5
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.dataset = self.load_dataset_with_retry()
        self.mel_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels)

    def load_dataset_with_retry(self):
        retries = 0
        while retries < self.max_retries:
            try:
                dataset = load_dataset(self.dataset_name, split=self.split, cache_dir=self.cache_dir)
                return dataset
            except Exception as e:
                # Retry khi server error hoặc ConnectionError
                if "503" in str(e) or "ConnectionError" in str(e):
                    print(f"Server error ({type(e).__name__}), retry {retries+1}/{self.max_retries} in {self.retry_delay}s...")
                    retries += 1
                    time.sleep(self.retry_delay)
                else:
                    raise e
        raise Exception(f"Failed to load dataset after {self.max_retries} retries.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["transcription"]
        audio_array = item["audio"]["array"]

        # Text -> indices
        text_indices = [char2idx.get(c, char2idx['<unk>']) for c in text]
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        # Audio -> mel
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        mel_spec = self.mel_transform(audio_tensor)  # [n_mels, T]

        # Chuẩn hóa (nếu đã tính mean, std)
        if self.mel_mean is not None and self.mel_std is not None:
            mel_spec = (mel_spec - self.mel_mean) / (self.mel_std + 1e-5)

        return text_tensor, mel_spec

################################################################
# collate_fn
################################################################
def collate_fn(batch):
    """
    batch: list of (text_tensor, mel_tensor)
    """
    texts, mels = zip(*batch)
    text_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    mel_lengths = torch.tensor([m.size(1) for m in mels], dtype=torch.long)

    max_text_len = text_lengths.max().item()
    max_mel_len = mel_lengths.max().item()

    # Pad text
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, :len(t)] = t

    # Pad mel
    padded_mels = torch.zeros(len(mels), mels[0].size(0), max_mel_len, dtype=torch.float32)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.size(1)] = m

    return padded_texts, padded_mels, text_lengths, mel_lengths
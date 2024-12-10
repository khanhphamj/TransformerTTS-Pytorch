import os
import torchaudio
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Optional, Dict, Any, List
from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_NAME, SAMPLE_RATE, N_MELS, NUM_WORKERS


def _process_single_example(audio_data: Dict[str, Any]):
    """
    Helper function để chuyển 1 audio path thành Mel spectrogram.
    """
    if isinstance(audio_data, dict) and "path" in audio_data:
        try:
            waveform, sample_rate = torchaudio.load(audio_data["path"])

            # Resample nếu cần
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            # Chuyển waveform sang Mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_mels=N_MELS
            )
            mel_spectrogram = mel_transform(waveform)
            return mel_spectrogram
        except Exception as e:
            print(f"Error processing audio {audio_data['path']}: {e}")
            return None
    else:
        print(f"Invalid audio data format: {audio_data}")
        return None

def preprocess_audio_factory(has_text: bool, text_column: Optional[str] = None):
    """
    Tạo hàm preprocess batch tùy thuộc vào việc có text column hay không.
    """
    def preprocess_audio(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        mel_spectrograms = []
        texts = [] if has_text else None

        if has_text and text_column is not None:
            for audio_data, text_data in zip(batch["audio"], batch[text_column]):
                mel_spec = _process_single_example(audio_data)
                mel_spectrograms.append(mel_spec)
                texts.append(text_data if text_data is not None else "")
            return {"mel_spectrogram": mel_spectrograms, text_column: texts}
        else:
            for audio_data in batch["audio"]:
                mel_spec = _process_single_example(audio_data)
                mel_spectrograms.append(mel_spec)
            return {"mel_spectrogram": mel_spectrograms}

    return preprocess_audio


def process_dataset(splits: Optional[List[str]] = None) -> DatasetDict:
    """
    Load raw dataset, tìm text column (nếu có), preprocess (chuyển audio -> mel),
    lưu dataset đã preprocess xuống đĩa.
    """
    if splits is None:
        splits = ["train"]

    # Load dataset từ đĩa
    raw_dataset_path = os.path.join(RAW_DATA_DIR, DATASET_NAME.replace("/", "_"))
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_dataset_path}. Please run load_raw_dataset_all_splits first.")

    dataset = load_from_disk(raw_dataset_path)

    # Đảm bảo dataset là DatasetDict
    if isinstance(dataset, Dataset):
        # Nếu chỉ có 1 split duy nhất
        dataset = DatasetDict({"train": dataset})
    else:
        # Dataset đã là DatasetDict
        pass

    # Lọc ra những split cần
    if splits:
        filtered = {sp: dataset[sp] for sp in splits if sp in dataset}
        dataset = DatasetDict(filtered)

    # Tìm text column
    sample_split = splits[0]
    possible_text_columns = ["text", "sentence", "transcription", "raw_text"]
    text_column = None
    for col in possible_text_columns:
        if col in dataset[sample_split].column_names:
            text_column = col
            break

    has_text = text_column is not None
    preprocess_fn = preprocess_audio_factory(has_text, text_column)

    # Các cột cần remove
    remove_cols = ["audio"]
    # Nếu bạn muốn giữ text thì không remove text_column, nếu không thì remove nó.

    processed_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=8,
        num_proc=NUM_WORKERS if NUM_WORKERS else 1,
        remove_columns=remove_cols
    )

    # Lưu dataset sau preprocess
    processed_path = os.path.join(PROCESSED_DATA_DIR, DATASET_NAME.replace("/", "_"))
    os.makedirs(processed_path, exist_ok=True)
    processed_dataset.save_to_disk(processed_path)
    print(f"Processed dataset saved to {processed_path}")

    return processed_dataset
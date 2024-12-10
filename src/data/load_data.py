import os
from datasets import load_dataset
from datasets import DatasetDict
from typing import List
from src.config.config import DATASET_NAME, RAW_DATA_DIR


def load_raw_dataset_all_splits(splits: List[str] = None):
    """
    Tải toàn bộ dataset với các splits chỉ định và lưu xuống ổ đĩa.
    Nếu splits = None, cố gắng tải tất cả splits mặc định (thường là train, test, validation).
    """
    if splits is None:
        # Nếu không truyền split, load_dataset sẽ tự động tải tất cả các split có sẵn.
        dataset = load_dataset(DATASET_NAME)
    else:
        # Tải riêng từng split rồi ghép lại
        dataset_dict = {}
        for sp in splits:
            ds = load_dataset(DATASET_NAME, split=sp)
            dataset_dict[sp] = ds
        dataset = DatasetDict(dataset_dict)

    # Tạo thư mục lưu trữ
    save_path = os.path.join(RAW_DATA_DIR, DATASET_NAME.replace("/", "_"))
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"Dataset {DATASET_NAME} (all requested splits) saved to {save_path}")
    return dataset


def load_raw_dataset_from_disk(splits: List[str] = None):
    """
    Load raw dataset từ ổ đĩa (sau khi đã tải và save trước đó).
    Nếu splits = None, tải toàn bộ datasetdict đã lưu.
    """
    load_path = os.path.join(RAW_DATA_DIR, DATASET_NAME.replace("/", "_"))
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No raw dataset found at {load_path}. Please run load_raw_dataset_all_splits first.")

    from datasets import load_from_disk
    dataset = load_from_disk(load_path)

    # Nếu cần, lọc ra chỉ các split mong muốn
    if splits:
        filtered = {sp: dataset[sp] for sp in splits if sp in dataset}
        dataset = DatasetDict(filtered)
    return dataset
import torch
from torch.utils.data import DataLoader
from src.data.dataset import (
    ISLWordDataset, 
    ISLSentenceDataset, 
    collate_fn_padd
)

def run_dataset_tests():
    print("--- [TEST 1] Testing ISLWordDataset (video) ---")
    try:
        word_video_dataset = ISLWordDataset(data_type='video')
        if len(word_video_dataset) == 0:
            print("WARNING: Word video dataset is empty! Check data/raw/Kaggle_Words/")
            return

        print(f"Found {len(word_video_dataset)} word video samples.")
        
        # Get the first sample
        video_sample, label_index = word_video_dataset[0]
        
        print(f"  - Single video sample shape: {video_sample.shape}")
        print(f"     (Should be [Time, Channels, Height, Width], e.g., [T, 3, 224, 224])")
        print(f"  - Single video label index: {label_index} (Type: {type(label_index)})")
        print("--- [TEST 1] PASSED --- \n")
        
    except Exception as e:
        print(f"--- [TEST 1] FAILED: {e} --- \n")

    print("--- [TEST 2] Testing ISLWordDataset (pose) ---")
    try:
        word_pose_dataset = ISLWordDataset(data_type='pose')
        if len(word_pose_dataset) == 0:
            print("WARNING: Word pose dataset is empty! Check data/raw/Kaggle_Words/")
            return

        print(f"Found {len(word_pose_dataset)} word pose samples.")
        
        # Get the first sample
        pose_sample, label_index = word_pose_dataset[0]
        
        print(f"  - Single pose sample shape: {pose_sample.shape}")
        print(f"     (Should be [Time, Features], e.g., [T, 258])")
        print(f"  - Single pose label index: {label_index} (Type: {type(label_index)})")
        print("--- [TEST 2] PASSED --- \n")
        
    except Exception as e:
        print(f"--- [TEST 2] FAILED: {e} --- \n")


    print("--- [TEST 3] Testing ISLSentenceDataset and DataLoader ---")
    try:
        sentence_dataset = ISLSentenceDataset()
        if len(sentence_dataset) == 0:
            print("WARNING: Sentence dataset is empty! Check data/raw/ISL_CSLTR/ISL_CSLTR.csv")
            return

        print(f"Found {len(sentence_dataset)} sentence samples.")
        print(f"  - Vocabulary size (with <BLANK>): {len(sentence_dataset.char_to_index)}")
        print(f"  - Char-to-Index Map (sample): {list(sentence_dataset.char_to_index.items())[:5]}...")
        
        # Test the DataLoader with our custom collate_fn
        # Using a small batch size of 2 for testing
        sentence_loader = DataLoader(
            sentence_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn_padd
        )
        
        # Get one batch from the loader
        data_batch, target_batch, data_lengths, target_lengths = next(iter(sentence_loader))
        
        print("\n  --- Testing DataLoader (Batch Size=2) ---")
        print(f"  - Padded data batch shape: {data_batch.shape}")
        print(f"     (Should be [2, T_max, C, H, W])")
        
        print(f"  - Padded target batch shape: {target_batch.shape}")
        print(f"     (Should be [2, L_max])")
        
        print(f"  - Video lengths in batch: {data_lengths}")
        print(f"     (Should be a tensor with 2 numbers, e.g., [50, 45])")
        
        print(f"  - Label lengths in batch: {target_lengths}")
        print(f"     (Should be a tensor with 2 numbers, e.g., [12, 9])")
        
        print("--- [TEST 3] PASSED --- \n")

    except Exception as e:
        print(f"--- [TEST 3] FAILED: {e} --- \n")

if __name__ == "__main__":
    run_dataset_tests()
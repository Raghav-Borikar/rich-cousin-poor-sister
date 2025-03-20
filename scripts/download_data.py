# scripts/download_data.py
import os
import json
from datasets import load_dataset, DatasetDict

def download_nllb_data():
    """Download and save NLLB Hindi-Chhattisgarhi parallel corpus"""
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load dataset from Hugging Face - only use the 'train' split
    print("Downloading NLLB dataset...")
    dataset = load_dataset(
        "allenai/nllb", 
        "hin_Deva-hne_Deva",
        split="train"  # Only request the train split
    )

    # Create train/val/test splits manually (80/10/10)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)
    
    splits = {
        "train": train_test["train"],
        "val": test_valid["train"],
        "test": test_valid["test"]
    }

    for split_name, split_data in splits.items():
        # Convert to required format
        processed = []
        for item in split_data:
            # Extract source and target from the 'translation' field
            processed.append({
                "source": item['translation']['hin_Deva'],
                "target": item['translation']['hne_Deva']
            })
        
        # Save to JSON
        output_path = f"data/processed/{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(processed)} {split_name} examples to {output_path}")

if __name__ == "__main__":
    download_nllb_data()

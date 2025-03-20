# scripts/download_data.py
import os
import json
from datasets import load_dataset

def download_nllb_data():
    """Download and save NLLB Hindi-Chhattisgarhi parallel corpus"""
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load dataset from Hugging Face
    print("Downloading NLLB dataset...")
    dataset = load_dataset(
        "allenai/nllb", 
        "hin_Deva-hne_Deva",
        split="train+validation+test"
    )

    # Process and save splits
    splits = {
        "train": dataset.train_test_split(test_size=0.2)['train'],
        "val": dataset.train_test_split(test_size=0.2)['test'].train_test_split(test_size=0.5)['train'],
        "test": dataset.train_test_split(test_size=0.2)['test'].train_test_split(test_size=0.5)['test']
    }

    for split_name, split_data in splits.items():
        # Convert to required format
        processed = []
        for item in split_data:
            processed.append({
                "source": item['translation']['source'],
                "target": item['translation']['target']
            })
        
        # Save to JSON
        output_path = f"data/processed/{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(processed)} {split_name} examples to {output_path}")

if __name__ == "__main__":
    download_nllb_data()

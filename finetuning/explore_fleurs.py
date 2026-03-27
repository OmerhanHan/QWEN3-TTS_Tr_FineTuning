from datasets import load_dataset
import json

# FLEURS veri kümesini yükle
fleurs = load_dataset("google/fleurs", "tr_tr", split="train", streaming=True)

# İlk birkaç örneği incele
for i, sample in enumerate(fleurs):
    print(f"Sample {i+1}:")
    print(json.dumps(sample, indent=2, default=str))
    print("-" * 50)
    
    if i >= 2:  # Sadece ilk 3 örneği göster
        break
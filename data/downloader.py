import os
from datasets import load_dataset

def download_hf_dataset_locally(dataset_name, target_location):

    print(f"--- Downloading dataset: {dataset_name} ---")
    
    dataset = load_dataset(dataset_name)
    
    if not os.path.exists(target_location):
        os.makedirs(target_location)
        print(f"Created directory: {target_location}")
    
    dataset.save_to_disk(target_location) # type: ignore
    
    print(f"Successfully saved '{dataset_name}' to: {target_location}")

download_hf_dataset_locally(
    dataset_name="cajcodes/political-bias", 
    target_location="./data/sample_data"
)
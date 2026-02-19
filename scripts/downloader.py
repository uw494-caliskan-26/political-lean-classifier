from datasets import load_dataset

# Configuration
DATASET_NAME = "pietrolesci/hyperpartisan_news_detection"
SAVE_PATH = "./data/data.parquet"
NUM_ROWS = 50000
COLUMNS_TO_KEEP = ["news_text", "title", "bias"] 

def download_subset():
    dataset = load_dataset(DATASET_NAME, split="train")
    
    dataset = dataset.shuffle(seed=42).select(range(NUM_ROWS)) # type: ignore
    
    columns = dataset.column_names
    columns_to_remove = [col for col in columns if col not in COLUMNS_TO_KEEP]
    final_ds = dataset.remove_columns(columns_to_remove)
    
    final_ds.to_parquet(SAVE_PATH)
    print("Done!")

if __name__ == "__main__":
    download_subset()
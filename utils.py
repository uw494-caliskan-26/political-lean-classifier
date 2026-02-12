import os
import pandas as pd
from pathlib import Path
from datasets import load_from_disk

def format_cajcodes_political_bias(input_path, target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    output_path = Path(target_path)  / 'data.csv'

    dataset = load_from_disk(input_path)
    df = pd.DataFrame(dataset['train'])  # type: ignore

    label_map = {
        0: 'left', 1: 'left', 2: 'center', 3: 'right', 4: 'right'
    }

    df['label'] = df['label'].map(label_map)
    df.to_csv(output_path, index=False)

    return output_path
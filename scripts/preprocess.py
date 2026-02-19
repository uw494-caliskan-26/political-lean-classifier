import pandas as pd
import trafilatura
import re

INPUT_PATH = './data/data.parquet'
OUTPUT_PATH = './data/clean_data.parquet'

def preprocess(input_parquet, output_parquet):
    df = pd.read_parquet(input_parquet)

    df['cleaned_news'] = df['news_text'].apply(clean_with_trafilatura)

    df['text'] = df['title'].fillna('') + "\n" + df['cleaned_news'] # type: ignore
    df['label'] = df['bias'].astype(float)
    
    df[['text', 'label']].to_parquet(output_parquet, index=False)
    print("Preprocessed with Trafilatura successfully.")

def clean_with_trafilatura(html_content):

    # use trafilatura to extract text from HTML content
    text = trafilatura.extract(html_content, favor_precision=False, include_tables=True)
    
    if not text:
        return ""

    # replace multiple consecutive dots with a single space
    text = re.sub(r'\.{2,}', ' ', text)
    text = ' '.join(text.split())

    # replce newlines and carriage returns with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')

    return text

if __name__ == "__main__":
    preprocess(INPUT_PATH, OUTPUT_PATH)
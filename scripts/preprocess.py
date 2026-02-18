import pandas as pd
import trafilatura
import re

INPUT_PATH = './data/data.csv'
OUTPUT_PATH = './data/final_clean.csv'

def advanced_clean_with_trafilatura(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    def clean_with_trafilatura(html_content):
        if not isinstance(html_content, str):
            return ""

        text = trafilatura.html2txt(html_content)
        
        if not text:
            return ""

        text = re.sub(r'\.{2,}', ' ', text)
        
        text = ' '.join(text.split())
        return text

    df['cleaned_news'] = df['news_text'].apply(clean_with_trafilatura)

    df['text'] = df['title'].fillna('') + " " + df['cleaned_news'] # type: ignore
    df['label'] = df['bias'].astype(float)
    
    df[['text', 'label']].to_csv(output_csv, index=False)
    print("Preprocessed with Trafilatura successfully.")

if __name__ == "__main__":
    advanced_clean_with_trafilatura(INPUT_PATH, OUTPUT_PATH)
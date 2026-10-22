from datasets import load_dataset
import re

def clean(t):
    t = re.sub(r"\{\{.*?\}\}", "", t)   # remove placeholders like {{Order Number}}
    return re.sub(r"\s+", " ", t).strip().lower()

def load():
    df = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"].to_pandas()
    df["text"] = df["instruction"].map(clean)
    return df

if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df["category"].value_counts())
    print(df["intent"].nunique())
    print(df[["text", "category", "intent"]].head(10))
import os
import random
import whisper
import string
import pandas as pd
from tqdm import tqdm
from typing import List
import jiwer 

random.seed(42)

DATA_DIR = "/Users/mac/Desktop/cs266-final-project/cv-corpus-21.0-delta-2025-03-14/en"
CLIP_DIR = os.path.join(DATA_DIR, "clips")
MODEL_SIZE = "base"

def tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.strip().split()

def load_and_sample(tsv_file: str, num_samples: int) -> pd.DataFrame:
    df = pd.read_csv(tsv_file, sep="\t")
    df = df[df["path"].notnull() & df["sentence"].notnull()]
    sampled_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    return sampled_df

def transcribe_clips(df: pd.DataFrame, clip_dir: str, model) -> List[str]:
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        audio_path = os.path.join(clip_dir, row["path"])
        try:
            result = model.transcribe(audio_path, language='en', fp16=False)
            predictions.append(result["text"])
        except Exception as e:
            print(f"Error transcribing {row['path']}: {e}")
            predictions.append("")
    return predictions

def compare_transcriptions(actuals: List[str], preds: List[str], paths: List[str]):
    mismatches = []
    for i, (a, p) in enumerate(zip(actuals, preds)):
        if tokenize(a) != tokenize(p):
            mismatches.append(f"Clip: {paths[i]}\nActual: {a}\nPredicted: {p}\n\n")
    return mismatches

def calculate_wer(actuals: List[str], preds: List[str]) -> float:
    """Calculate Word Error Rate (WER)"""
    return jiwer.wer(actuals, preds)

def calculate_token_match_rate(actuals: List[str], preds: List[str]) -> float:
    """Calculate Token Match Rate"""
    matches = 0
    total = len(actuals)
    for a, p in zip(actuals, preds):
        if tokenize(a) == tokenize(p):
            matches += 1
    return matches / total

def main():
    print("Loading model")
    model = whisper.load_model(MODEL_SIZE)

    print("Loading validated, invalidated and other datasets")
    validated_df = pd.read_csv(os.path.join(DATA_DIR, "validated.tsv"), sep="\t")
    invalidated_df = pd.read_csv(os.path.join(DATA_DIR, "invalidated.tsv"), sep="\t")
    other_df = pd.read_csv(os.path.join(DATA_DIR, "other.tsv"), sep="\t")

    df = pd.concat([validated_df, invalidated_df, other_df], ignore_index=True)
    df = df[df["path"].notnull() & df["sentence"].notnull() & df["age"].notnull()]

    valid_ages = ["teens", "twenties", "thirties", "forties", "fifties", "sixties"]
    df = df[df["age"].isin(valid_ages)]

    summary = []

    for age in valid_ages:
        print(f"\nProcessing age group: {age}")
        group_df = df[df["age"] == age]

        if len(group_df) < 100:
            print(f"Not enough samples for age group '{age}', skipping")
            print(f"length: {len(group_df)}")
            continue

        group_df = group_df.sample(n=100, random_state=42).reset_index(drop=True)

        predicted_transcripts = transcribe_clips(group_df, CLIP_DIR, model)

        mismatches = compare_transcriptions(group_df["sentence"].tolist(), predicted_transcripts, group_df["path"].tolist())
        
        mismatch_file = f"mismatches_{age}.txt"
        with open(mismatch_file, "w") as f:
            f.writelines(mismatches)
        print(f"Mismatches saved to '{mismatch_file}'.")

        wer = calculate_wer(group_df["sentence"].tolist(), predicted_transcripts)
        token_match_rate = calculate_token_match_rate(group_df["sentence"].tolist(), predicted_transcripts)

        print(f"Word Error Rate (WER): {wer * 100:.2f}%")
        print(f"Token Match Rate: {token_match_rate * 100:.2f}%")

        summary.append((age, len(group_df), wer, token_match_rate))

    with open("age_group_summary.csv", "w") as f:
        f.write("Age Group,Num Clips,WER,Token Match Rate\n")
        for row in summary:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f}\n")
    
    print("\n Summary saved to 'age_group_summary.csv'")


if __name__ == "__main__":
    main()

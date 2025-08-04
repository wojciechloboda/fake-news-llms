import subprocess
import time
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import argparse

BASE_PROMPT = (
    "Carefully read the following news article and assess whether it is likely to be FAKE or REAL "
    "based on its semantic content, coherence, plausibility, and tone. "
    "Avoid guessing based on keywords—use reasoning. "
    "Respond with only one word: FAKE or REAL.\n\n"
)

def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.RequestException:
        return False
    
def start_ollama_server():
    process = subprocess.Popen(
        ["ollama", "server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("Starting Ollama server...")
    time.sleep(5)  
    return process

import subprocess

def pull_model(model_name):
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Print the output from the command
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print("Error Output:")
            print(result.stderr.strip())
        
        if result.returncode == 0:
            print(f"Model '{model_name}' is ready.")
        else:
            print(f"Error pulling model '{model_name}': {result.stderr.strip()}")
    except Exception as e:
        print(f"Exception pulling model: {e}")


def ask_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[No response received]")
    except requests.RequestException as e:
        return f"Error communicating with Ollama: {e}"
    
def evaluate_classification(df_true, df_pred, model_name, output_dir="metrics"):
    os.makedirs(output_dir, exist_ok=True)

    df_merged = pd.merge(df_true, df_pred, on='id', suffixes=('_true', '_pred'))
    df_merged = df_merged[df_merged['label_pred'] != -1]

    y_true = df_merged['label']
    y_pred = df_merged['label_pred']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"Model: {model_name}")
    print("Classification Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake (0)", "Real (1)"]))

    metrics_data = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    df_metrics = pd.DataFrame([metrics_data])

    filename = f"{output_dir}/metrics_{model_name}.csv"
    df_metrics.to_csv(filename, index=False)

def response_to_label(response: str) -> int:
    response = response.strip().upper()
    if response == "REAL":
        return 1
    elif response == "FAKE":
        return 0
    else:
        return -1  # Unknown or unusable


def evaluate_classification(df_true: pd.DataFrame, df_pred: pd.DataFrame, model_name: str):
    df_merged = pd.merge(df_true, df_pred, on='id', suffixes=('_true', '_pred'))
    df_merged = df_merged[df_merged['label_pred'] != -1]

    y_true = df_merged['label']
    y_pred = df_merged['label_pred']

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print("\n📊 Classification Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    # Save to metrics CSV
    metrics_df = pd.DataFrame([{
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }])
    os.makedirs("metrics", exist_ok=True)
    metrics_df.to_csv(f"metrics/{model_name}_metrics.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="Run model prediction and evaluation.")
    parser.add_argument("--model", required=True, help="Name of the model (e.g., mistral)")
    parser.add_argument("--dataset", required=True, help="Prefix of the dataset (e.g., buzzfeed)")

    args = parser.parse_args()
    model_name = args.model
    dataset_prefix = args.dataset

    content_path = f"{dataset_prefix}_content.csv"
    labels_path = f"{dataset_prefix}_labels.csv"

    server_process = None
    if not is_ollama_running():
        server_process = start_ollama_server()

    pull_model(model_name)

    df_content = pd.read_csv(content_path)
    df_true = pd.read_csv(labels_path)

    predictions = []
    for i, row in df_content.iterrows():
        prompt = BASE_PROMPT + row['content']
        print(f"\nQuerying id {row['id']}")
        response = ask_ollama(prompt, model=model_name)
        label = response_to_label(response)
        print(f"Answer: {response} -> Label: {label}")
        predictions.append(label)

    df_pred = pd.DataFrame({
        'id': df_content['id'],
        'label_pred': predictions
    })

    evaluate_classification(df_true, df_pred, model_name)
    
    if server_process is not None:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()








# Fake News Detection Using Local LLMs

## Setup

1. Install Ollama from https://ollama.com/download
2. Install Python dependencies:

`pip install -r requirements.txt`

---

## Usage

Prepare Dataset

The dataset should consist of two CSV files in the same directory:

- {NAME}_content.csv with columns: id, content
- {NAME}_labels.csv with columns: id, label (0 = fake, 1 = real)

You can use the example script to prepare datasets:

`python download_dataset.py`

---

## Run Classification and Evaluation

`python classify_and_evaluate.py --model MODEL_NAME --dataset DATASET_PATH`

- MODEL_NAME: Ollama model name (e.g., llama3.1)
- DATASET_PATH: path prefix of dataset files (e.g., datasets/buzzfeed)

---

Example

`python classify_and_evaluate.py --model llama3.1 --dataset datasets/buzzfeed`

Dataset folder structure:

```
datasets/
 ├─ buzzfeed_content.csv
 └─ buzzfeed_labels.csv
```
---

Output

- Creates a DataFrame with columns: [id, label_pred]
- label_pred values:
  - 0 = FAKE
  - 1 = REAL
  - -1 = Unable to parse model output as FAKE or REAL

- Saves classification metrics to:
  metrics/{MODEL_NAME}_metrics.csv


###  Plotting Metrics

To compare performance across models, run:

`python plot_all.py`

This script visualizes metrics from all CSV files in the "metrics" directory. 

---

Example of Unparsable Output (Label -1)

The article's source is not identified, and a search for the provided Twitter links reveals that the accounts do not exist or the tweets were never posted... (and so forth)

---

Example Ollama Models

See Ollama Model Library (https://ollama.com/library) for full list, e.g.:

- llama3.1
- deepsekk
- gemma3n
...

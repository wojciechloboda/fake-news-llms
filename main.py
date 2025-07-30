
# Load fake and true here 

# import kagglehub
# from kagglehub import KaggleDatasetAdapter


# file_path = "News _dataset/Fake.csv"

# df = kagglehub.load_dataset(
#   KaggleDatasetAdapter.PANDAS,
#   "emineyetm/fake-news-detection-datasets",
#   file_path,
# )

# print("First 5 text values:", df["text"].head().tolist())

# Dataset must be refined, routers prefixes must be removed 
# https://www.kaggle.com/datasets/mdepak/fakenewsnet/data?select=BuzzFeed_real_news_content.csv <- better dataset
# https://arxiv.org/pdf/1809.01286


import subprocess
import time
import requests

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

def pull_model(model_name):
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
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
    

model_list = ['mistral']    

if __name__ == "__main__":
    server_process = None
    if not is_ollama_running():
        server_process = start_ollama_server()

    model_name = model_list[0]
    pull_model(model_name)

    query = input("Enter your question: ")
    answer = ask_ollama(query, model=model_name)
    print("Answer:", answer)

    if server_process is not None:
        server_process.terminate()
        server_process.wait()










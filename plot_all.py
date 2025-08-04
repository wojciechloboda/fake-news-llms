import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_individual_metrics(metrics_folder="metrics", output_dir="metric_plots"):
    os.makedirs(output_dir, exist_ok=True)

    dfs = []
    for f in os.listdir(metrics_folder):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(metrics_folder, f)))
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.set_index('model', inplace=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    for metric in metrics:
        ax = df_all[[metric]].plot(kind='bar', legend=False, figsize=(8, 5), color='skyblue')
        plt.title(f"{metric.capitalize()} Comparison")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()

        file_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(file_path)
        plt.close()

plot_individual_metrics()        

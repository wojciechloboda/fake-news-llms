import os
import kagglehub
import pandas as pd

OUTPUT_DIR = "datasets"
ARTICLE_COUNT = 100
DATASET_NAME = "buzzfeed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

path = kagglehub.dataset_download("mdepak/fakenewsnet")

fake_path = f'{path}/BuzzFeed_fake_news_content.csv'
real_path = f'{path}/BuzzFeed_real_news_content.csv'

df_fake = pd.read_csv(fake_path)[['text']].copy()
df_fake['label'] = 0  # Fake news

df_real = pd.read_csv(real_path)[['text']].copy()
df_real['label'] = 1  # Real news

df_combined = pd.concat([df_fake, df_real], ignore_index=True).dropna(subset=['text'])
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

df_combined = df_combined.head(ARTICLE_COUNT)
df_combined['id'] = df_combined.index

content_df = df_combined[['id', 'text']].rename(columns={'text': 'content'})
labels_df = df_combined[['id', 'label']]

content_df.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_content.csv"), index=False)
labels_df.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_labels.csv"), index=False)

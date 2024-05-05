import pandas as pd
import random
import nltk

# Download NLTK data if not already present
nltk.download('punkt')

# Load the TSV file into a DataFrame
df_2023 = pd.read_csv('./NADI2023_TWT.tsv', sep='\t')
df_2021 = pd.read_csv('./NADI2021-TWT.tsv', sep='\t')
df_2020 = pd.read_csv('./NADI2020-TWT.tsv', sep='\t')
df_kaggle = pd.read_csv('./messages.csv')

corpus_2023 = df_2023['#2_content'].tolist()
corpus_2021 = df_2021['#2_tweet_content'].tolist()
corpus_2020 = df_2020['#2_tweet_content'].tolist()

corpus_kaggle = df_kaggle['sentence'].tolist()

# Tokenize each sentence and calculate the total number of tokens
total_tokens = 0
for sentence in corpus_2023:
    tokens = nltk.word_tokenize(sentence)
    total_tokens += len(tokens)

for sentence in corpus_2021:
    tokens = nltk.word_tokenize(sentence)
    total_tokens += len(tokens)

for sentence in corpus_2020:
    tokens = nltk.word_tokenize(sentence)
    total_tokens += len(tokens)


for sentence in corpus_kaggle:
    tokens = nltk.word_tokenize(str(sentence))
    total_tokens += len(tokens)

print("Total number of tokens:", total_tokens)
print("Total number of sentences:", len(corpus_2023) + len(corpus_2021) + len(corpus_2020) + len(corpus_kaggle))
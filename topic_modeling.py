from bertopic import BERTopic
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

arabic_dialect_dataset = pd.read_csv('./arabic_dialect.csv')
print(arabic_dialect_dataset.head())

arabic_dialect_dataset.dropna(subset=['sentence'], inplace=True)


embedding_model = SentenceTransformer("aubmindlab/bert-base-arabertv02-twitter")

print("Embedding model loaded successfully")

topic_model = BERTopic(language="arabic", low_memory=True ,calculate_probabilities=True,
                     embedding_model=embedding_model, nr_topics=20, verbose=True )

print("Topic model loaded successfully")

# embeddings = embedding_model.encode(arabic_dialect_dataset['sentence'].tolist(), show_progress_bar=True)


embeddings = np.load('./embeddings.npy')

print("Embeddings generated successfully")

topics, probs = topic_model.fit_transform(arabic_dialect_dataset['sentence'], embeddings = embeddings)

print("Topics and probabilities generated successfully")

topic_probs = pd.DataFrame(probs)

topic_probs.columns = [f'topic_{index}' for index in range(len(topic_probs.columns))]

print(topic_probs.head())

arabic_dialect_dataset = pd.concat([arabic_dialect_dataset, topic_probs], axis=1)

arabic_dialect_dataset.to_csv('./arabic_dialect_with_topic_probs.csv', index=False)


print("Topic probabilities saved successfully")
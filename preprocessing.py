import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from arabert.preprocess import ArabertPreprocessor

import re

messages_df = pd.read_csv('./messages.csv')
dialect_dataset_df = pd.read_csv('./dialect_dataset.csv')

messages_df.dropna(inplace=True)
dialect_dataset_df.dropna(inplace=True)

messages_df['id'] = messages_df['id'].astype(np.int64)
dialect_dataset_df['id'] = dialect_dataset_df['id'].astype(np.int64)

merged_df = pd.merge(messages_df, dialect_dataset_df, on='id')


encoder = OneHotEncoder()
encoded_dialect = encoder.fit_transform(merged_df[['dialect']])

# Create a new DataFrame with the encoded dialect column
encoded_df = pd.DataFrame(encoded_dialect.toarray(), columns=encoder.get_feature_names_out(['dialect']))

# Concatenate the encoded DataFrame with the original merged DataFrame
merged_df = pd.concat([merged_df, encoded_df], axis=1)

merged_df['user'] = merged_df['sentence'].apply(lambda x: re.findall(r'@(\w+)', x)[0] if re.findall(r'@(\w+)', x) else None)
merged_df['sentence'] = merged_df['sentence'].str.split().str[1:].str.join(' ')

merged_df['sentence'] = merged_df['sentence'].apply(lambda x: re.sub(r'@\w+|http\S+', '', x))
merged_df['sentence'] = merged_df['sentence'].apply(lambda x: re.sub(r'[a-zA-Z]', '', x))




model_name="aubmindlab/bert-base-arabertv02-twitter"
arabert_prep = ArabertPreprocessor(model_name=model_name)

merged_df['sentence'] = merged_df['sentence'].apply(arabert_prep.preprocess)

merged_df.dropna(subset=['sentence'], inplace=True)

region_mapping = {'AE': "GULF",
       'BH': "GULF", 
       'KW': "GULF", 
       'OM':"GULF",
       'QA':"GULF", 
       'SA':"GULF",
       'YE':"GULF",
       'SD':"NILE BASIN", 
       'EG': "NILE BASIN", 
       'IQ': "LEVANT", 
       'JO': "LEVANT",
       'LB':"LEVANT", 
       'PL':"LEVANT", 
       'SY':"LEVANT",
       'DZ': "MAGHREB", 
       'LY':"MAGHREB", 
       'MA':"MAGHREB", 
       'TN':"MAGHREB", 
}

merged_df['region'] = merged_df['dialect'].map(region_mapping)

column_order = ['id', 'user', 'sentence', 'region', 'dialect']
remaining_columns = [col for col in merged_df.columns if col not in column_order]
new_column_order = column_order + remaining_columns

merged_df = merged_df.reindex(columns=new_column_order)


num_unique_users = merged_df['user'].nunique()
print("Number of unique users:", num_unique_users)

num_dialects = merged_df['dialect'].nunique()
print("Number of dialects:", num_dialects)

num_regions = merged_df['region'].nunique()
print("Number of regions:", num_regions)

encoded_dialect = encoder.fit_transform(merged_df[['region']])

# Create a new DataFrame with the encoded dialect column
encoded_df = pd.DataFrame(encoded_dialect.toarray(), columns=encoder.get_feature_names_out(['region']))

# Concatenate the encoded DataFrame with the original merged DataFrame
merged_df = pd.concat([merged_df, encoded_df], axis=1)


merged_df.to_csv('./arabic_dialect.csv', index=False)

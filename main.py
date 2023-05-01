import openai
import pandas as pd
import numpy as np
import os
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

openai.api_key = os.getenv('OPENAI_API_KEY')

# use sentence_formatter to generate CSV from text
df = pd.read_csv('chapter1-sunset.csv')

if not os.path.exists('word_embeddings.csv'):
    print('generating embeddings for book')
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    df.to_csv('word_embeddings.csv')

print('loading embeddings')
df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

search_term = "romantic relationships ending"
print(f'getting embeddings for search term: {search_term}')
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

print('finding sentences similar to search term')
df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

print('top 20 results:')
print(df.sort_values("similarities", ascending=False).head(20))

# TODO take output and prompt chatgpt, "Given the sentence, extract the 'answer' for the search term"

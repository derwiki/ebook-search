import openai
import pandas as pd
import numpy as np
import os
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

openai.api_key = os.getenv('OPENAI_API_KEY')
# TODO convert lessistlost.txt to lessislost.csv
#   split by sentences
df = pd.read_csv('lessislost.csv')
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('word_embeddings.csv')
df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)

search_term = "Arthur's age"
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
df.sort_values("similarities", ascending=False).head(20)

# TODO take output and prompt chatgpt, "Given the sentence, extract the 'answer' for the search term"

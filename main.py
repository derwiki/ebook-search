import sys
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd
import numpy as np
import os
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

from sentence_formatter import text_to_sentence_csv

openai.api_key = os.getenv('OPENAI_API_KEY')


def apply_get_embedding(text):
    result = get_embedding(text, engine='text-embedding-ada-002')
    print(f'finished embedding')
    return result


def semantic_search(text_filename: str, search_term: str):
    if not text_filename.endswith('.txt'):
        raise Exception(f'Invalid input text file: {text_filename}')

    # use sentence_formatter to generate CSV from text
    csv_filename = text_to_sentence_csv(text_filename)
    df = pd.read_csv(csv_filename)
    text_file_prefix = text_filename.replace('.txt', '')
    embeddings_filename = f'{text_file_prefix}_word_embeddings.csv'

    if not os.path.exists(embeddings_filename):
        print('generating embeddings for book')
        with ThreadPoolExecutor(max_workers=64) as executor:
            embeddings = list(executor.map(apply_get_embedding, df['text']))
        df['embedding'] = embeddings
        df.to_csv(embeddings_filename)

    print('loading embeddings')
    df = pd.read_csv(embeddings_filename)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    print(f'getting embeddings for search term: {search_term}')
    search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

    print('finding sentences similar to search term')
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

    print('top 20 results:')
    print(df.sort_values("similarities", ascending=False).head(20))

    # TODO take output and prompt chatgpt, "Given the sentence, extract the 'answer' for the search term"


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Missing required params")

    semantic_search(text_filename=sys.argv[1], search_term=sys.argv[2])

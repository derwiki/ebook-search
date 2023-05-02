import csv
import os.path
import re

from nltk.tokenize import sent_tokenize
import nltk
from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def text_to_sentence_csv(input_text_file: str) -> str:
    """
    Converts text to CSV with one line per sentence.
    Discards sentences that exceed the get_embedding token limit.

    :param input_text_file:
    :return: string value of the path for the output CSV file
    """
    if not os.path.exists(input_text_file):
        raise Exception(f'File not found: {input_text_file}')

    output_csv_filename = input_text_file.replace(".txt", ".csv")
    if os.path.exists(output_csv_filename):
        print('CSV already exists, reusing')
        return output_csv_filename

    print('Download the required NLTK data')
    nltk.download('punkt')

    print('Read the ebook text file')
    with open(input_text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    print('Discard superfluous line breaks')
    text = re.sub(r'\n(?:\s*\n)*(?=[a-z])', ' ', text)

    print('Tokenize the text into sentences')
    sentences = sent_tokenize(text)
    print(f'Found {len(sentences)} sentences')

    print('Write the sentences to a CSV file')
    with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csvfile.write('text\n')
        for sentence in sentences:
            token_ids = tokenizer.encode(sentence, return_tensors=None)
            token_count = len(token_ids)

            # Check if the token count is greater than the model's maximum context length
            if token_count > 8191:
                print(f"Skipping text with {token_count} tokens, which exceeds the model's maximum context length.")
            else:
                # sentences that set up dialog have newlines; strip them
                writer.writerow([sentence.replace('\n', ' ')])
    return output_csv_filename

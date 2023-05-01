import csv
import re

from nltk.tokenize import sent_tokenize
import nltk

print('Download the required NLTK data')
nltk.download('punkt')

print('Read the ebook text file')
with open('chapter1-sunset.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print('Discard superfluous line breaks')
text = re.sub(r'\n(?:\s*\n)*(?=[a-z])', ' ', text)
# text = re.sub(r'\n(?=[a-z])', ' ', text)

print('Tokenize the text into sentences')
sentences = sent_tokenize(text)

print('Write the sentences to a CSV file')
with open('chapter1-sunset.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csvfile.write('text\n')
    for sentence in sentences:
        # sentences that set up dialog have newlines; strip them
        writer.writerow([sentence.replace('\n', ' ')])

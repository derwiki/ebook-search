# Semantic Search using OpenAI

This Python program utilizes the OpenAI API to perform semantic search on a text file. It accepts two arguments: the name of the text file to search, and the search term. It then generates embeddings for each sentence in the text, compares those embeddings to the search term's embedding, and returns the top 20 matches sorted by similarity.


## Overview
This application will take an ebook and build a semantic search engine over the contents.

The code is all cribbed from the super helpful video 
[5. OpenAI Embeddings API - Searching Financial Documents](https://www.youtube.com/watch?v=xzHhZh7F25I)
by Part Time Larry.
* Specifically, the code came from this [iPython Notebook](https://colab.research.google.com/drive/1tttDqgnWL9yJtmlOFXJqA-BjQ1Pyfpax?usp=sharing#scrollTo=bUPM0-8iLNK0).


## Usage

To run the program, make sure you have the OpenAI API key in the environment variable `OPENAI_API_KEY`. Then simply pass in the text file name and search term as arguments like so:

```
python main.py ebook.txt "search term"
```

```
(venv) ➜ git:(main) ✗ python main.py lessislost-chapter1.txt "romantic relationships ending"
generating embeddings for book
loading embeddings
getting embeddings for search term: romantic relationships ending
finding sentences similar to search term
top 20 results:
     Unnamed: 0                                               text                                          embedding  similarities
696         696                 Just like that, the love was over.  [-0.0064529250375926495, -0.015399391762912273...      0.836687
369         369  In the middle of their time together as a coup...  [0.004660928621888161, -0.019074782729148865, ...      0.818846
543         543  I’m so sorry.”  I explained that not every rel...  [-0.030615320429205894, 0.000493743282277137, ...      0.812686
692         692  Twenty years a couple, they suddenly announced...  [-0.01228040549904108, -0.006871253717690706, ...      0.808795
86           86  The famous poet Robert Brownburn; easy for him...  [-0.02290256693959236, -0.007946318946778774, ...      0.807237
27           27            Doves within a cage cooed romantically.  [-0.029864875599741936, -0.026972133666276932,...      0.805637
373         373                           They were still in love.  [-0.0003517003497108817, -0.010070735588669777...      0.802216
0             0  LESS should have known, at the clinic a few we...  [-0.00749213295057416, 0.006945409812033176, 0...      0.801186
288         288  And with Less, she is talking about his and Ro...  [-0.03143167123198509, 0.002833976410329342, 0...      0.799768
526         526  Back when Less did not know if he deserved to ...  [-0.0036562231834977865, 0.0002235552383353933...      0.796240
18           18  This clumsiness of the heart also became appar...  [-0.0009649221901781857, 0.002075009746477008,...      0.796202
42           42                    Nine months of unmarital bliss.  [-0.0063613178208470345, -0.020452434197068214...      0.795985
700         700  “We took each other as far as we could.” Is th...  [-0.005477485246956348, -0.007059574127197266,...      0.794903
214         214  Marian looks still lively but despairing, like...  [-0.03386417776346207, -0.029376782476902008, ...      0.792012
362         362  Less remembers evenings up in a cabin, card ga...  [-0.0005376689950935543, -0.0115670096129179, ...      0.790500
613         613  The reception has ended, the Russian River Sch...  [-0.02422231063246727, -0.0009064676123671234,...      0.790403
423         423  Robert is being moved into his private grotto ...  [-0.018788842484354973, -0.008001681417226791,...      0.788506
46           46  Less never pressed for more than a kiss goodby...  [-0.016261950135231018, -0.01728760078549385, ...      0.788178
694         694                   They drank champagne and parted.  [0.0033095034305006266, -0.01542810257524252, ...      0.788084
475         475                          We’d just started dating.  [-0.02588159404695034, -0.013313786126673222, ...      0.787115
```

The program will generate embeddings for the sentences in the text file if they don't already exist, and then perform the semantic search on those embeddings. The output will be the top 20 matches sorted by similarity, along with their corresponding sentences.

## Dependencies

The program requires the installation of the following Python packages:
- pandas
- numpy
- openai

These can be installed using pip:

```
pip install pandas numpy openai
```

Additionally, the program relies on a custom module called `sentence_formatter`, which is included in the repository.

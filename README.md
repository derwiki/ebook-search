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
(venv) ➜  ebook-search git:(main) ✗ time venv/bin/python main.py lessislost.txt "alcoholic beverages" ; say done
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Download the required NLTK data
[nltk_data] Downloading package punkt to /Users/adam/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
Read the ebook text file
Discard superfluous line breaks
Tokenize the text into sentences
Found 10662 sentences
Write the sentences to a CSV file
Token indices sequence length is longer than the specified maximum sequence length for this model (12369 > 1024). Running this sequence through the model will result in indexing errors
Skipping text with 12369 tokens, which exceeds the model's maximum context length.
generating embeddings for book
500 embeddings processed.
1000 embeddings processed.
...
10000 embeddings processed.
10500 embeddings processed.
loading embeddings
getting embeddings for search term: alcoholic beverages
finding sentences similar to search term
top 20 results:
      Unnamed: 0                                               text                                          embedding  similarities
289          289                     And remember to drink alcohol.  [0.010054944083094597, -0.010920019820332527, ...      0.842849
2795        2795   When it comes to drinking, should we know no no?  [0.020439572632312775, -0.021846452727913857, ...      0.819814
3744        3744  With their beers and snacks on their laps, eat...  [0.01141290832310915, -0.018857283517718315, 0...      0.816803
1589        1589                             They were quite drunk.  [0.002823491347953677, -0.012792474590241909, ...      0.815624
3871        3871                   The other gets Southern Comfort.  [0.008989846333861351, -0.017885204404592514, ...      0.808843
7249        7249  Drinking even a small amount of hand sanitizer...  [0.03205445036292076, 0.0026009646244347095, 0...      0.808286
9897        9897  Addictions to alcohol overindulgence and drugs...  [0.014888299629092216, -0.005606885068118572, ...      0.808227
1583        1583  The other arrived late, in a light blue sweate...  [-0.0009539441671222448, -0.02703011780977249,...      0.804221
2593        2593  My ex-husband, for one.” She reemerges with a ...  [0.004508504644036293, -0.02552058733999729, -...      0.803345
873          873  Less manages to make himself a mini-cocktail b...  [-0.012840939685702324, -0.016079911962151527,...      0.803327
581          581                                   Another bourbon.  [-0.004019715823233128, -0.008585800416767597,...      0.801514
838          838  An offer of wine and Less shivers at the impos...  [0.005081620067358017, -0.03446187824010849, -...      0.800690
1223        1223  We had apparently brought with us only the cas...  [0.02989587001502514, -0.01232936792075634, 0....      0.799524
3329        3329  I made my way through the academics looking fo...  [0.006479981821030378, -0.026723962277173996, ...      0.798036
2637        2637  In curly dark hair and red glasses, but surely...  [-0.003154703648760915, -0.025602824985980988,...      0.797717
2564        2564  She points her finger at our hero to make one ...  [-0.00880262441933155, -0.02387617528438568, -...      0.797695
2288        2288  Less finds himself being led down the road to ...  [-0.011682722717523575, -0.007981733419001102,...      0.797487
1230        1230  For four cans of Dewey beer and assistance in ...  [0.014380093663930893, 0.004230041988193989, 0...      0.796130
3692        3692  Rebecca is wrapped in a blanket and has brough...  [0.007203337736427784, -0.018650192767381668, ...      0.794228
1581        1581  Let us savor the scene in all its details: The...  [-0.0007753131212666631, -0.010730543173849583...      0.794010
venv/bin/python main.py lessislost.txt "alcoholic beverages"  63.13s user 9.61s system 26% cpu 4:35.34 total
```

The program will generate embeddings for the sentences in the text file if they don't already exist, and then perform the semantic search on those embeddings. The output will be the top 20 matches sorted by similarity, along with their corresponding sentences.

## Dependencies

The program requires the installation of the following Python packages:
- matplotlib
- nltk
- numpy
- openai
- pandas
- plotly
- ratelimit
- scipy
- scikit-learn


These can be installed using pip:

```
pip install matplotlib nltk numpy openai pandas plotly ratelimit scipy scikit-learn
```

Additionally, the program relies on a custom module called `sentence_formatter`, which is included in the repository.

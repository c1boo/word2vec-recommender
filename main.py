import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot

# df = pd.read_excel('data.csv', sheet_name = 'Sheet1').drop(['Unnamed: 0'], axis = 1)
df = pd.read_csv("data.csv")


# Function for removing ASCII characters
def _remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)


# Function for converting to lower case
def make_lower_case(text):
    return text.lower()


# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


# Function for removing html
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


# Text Processing
df['Cleaned'] = df['overview'].apply(_remove_non_ascii)
df['Cleaned'] = df.Cleaned.apply(func=make_lower_case)
df['Cleaned'] = df.Cleaned.apply(func=remove_stop_words)
df['Cleaned'] = df.Cleaned.apply(func=remove_punctuation)
df['Cleaned'] = df.Cleaned.apply(func=remove_html)

# Splitting the description into words
corpus = []
for words in df['Cleaned']:
    corpus.append(words.split())

# ############# Training ##################
EMBEDDING_FILE = 'word2vec-google-news-300.gz'
google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Training our corpus with the model
google_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
google_model.build_vocab(corpus)
google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)


def vectors(x):
    # Creating a list for storing the vectors ('Description' into vectors)
    global word_embeddings
    word_embeddings = []

    # Reading the each 'Description'
    for line in df['Cleaned']:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in google_model.wv.key_to_index:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_model.wv[word]
                else:
                    avgword2vec = avgword2vec + google_model.wv[word]

        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
            word_embeddings.append(avgword2vec)


# Recommending the Top 5 similar movies
def recommendations(movie):
    # Calling the function vectors
    vectors(df)

    # Finding cosine similarity for the vectors
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    # Taking the Title and Movie Image Link and store in new dataframe called 'movies'
    movies = df[['title', 'img_url']]

    # Reverse mapping of the index
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    try:
        idx = indices[movie]
    except KeyError:
        print(f"No item with name:{movie} available.")
        return

    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recommend = movies.iloc[movie_indices]

    for index, row in recommend.iterrows():
        response = requests.get(row['img_url'])
        img = Image.open(BytesIO(response.content))
        plt.figure()
        plt.imshow(img)
        plt.title(row['title'])
        plt.show()
        print(row['title'])


while True:
    title = input("Enter title: ")
    print("Recommendations:")
    recommendations(title)

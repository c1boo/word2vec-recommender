import warnings
import PIL
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors

nltk.download('stopwords')
warnings.filterwarnings('ignore')

df = pd.read_excel('data.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)


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
df['Cleaned'] = df['Description'].apply(_remove_non_ascii)
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
# Building the TF-IDF model and calculating the TF-IDF score
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=5, stop_words='english')
tfidf.fit(df['Cleaned'])

# Getting the words from the TF-IDF model
tfidf_list = dict(zip(tfidf.get_feature_names_out(), list(tfidf.idf_)))

# TF-IDF words/column names
tfidf_feature = tfidf.get_feature_names_out()

# Storing the TFIDF Word2Vec embeddings
tfidf_vectors = []
line = 0

# For each 'Description'
for desc in corpus:

    # Word vectors are of zero length (using 300 dimensions)
    sent_vec = np.zeros(300)

    # Number of words with a valid vector in the 'Description'
    weight_sum = 0

    # For each word in the 'Description'
    for word in desc:
        if word in google_model.wv.key_to_index and word in tfidf_feature:
            vec = google_model.wv[word]
            tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_vectors.append(sent_vec)
    line += 1


# Recommending top 5 similar movies
def recommendations(movie):
    # Finding cosine similarity for the vectors
    cosine_similarities = cosine_similarity(tfidf_vectors, tfidf_vectors)

    # Taking the Title and Image Link and store in new data frame called movies
    movies = df[['Movie', 'ImgLink']]

    # Reverse mapping of the index
    indices = pd.Series(df.index, index=df['Movie']).drop_duplicates()

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

    scores = [x[1] for x in sim_scores]
    counter = 0
    for index, row in recommend.iterrows():
        response = requests.get(row['ImgLink'])

        plt.figure()
        try:
            img = Image.open(BytesIO(response.content))
            plt.imshow(img)
        except PIL.UnidentifiedImageError:
            continue

        plt.title(f"{row['Movie']} - Score: {scores[counter]:.4f}")
        counter += 1
        plt.show()
        print(row['Movie'])


while True:
    title = input("Enter title: ")
    print("Recommendations:")
    recommendations(title)

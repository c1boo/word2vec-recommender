import warnings
import PIL
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
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


# ASCII karakterleri dışındaki karakterleri kaldırma
def _remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)


# Tüm harfleri küçük harfe çevirme
def make_lower_case(text):
    return text.lower()


# Stop wordleri kaldırma
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if w not in stops]
    text = " ".join(text)
    return text


# HTML etiketlerini kaldırma
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


# Noktalama işaretlerini kaldırma
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

# Açıklamayı kelimelere bölere liste oluşturma
corpus = []
for words in df['Cleaned']:
    corpus.append(words.split())

# ############# Google News Word2Vec modelini yükleme ve eğitme ##################
EMBEDDING_FILE = 'word2vec-google-news-300.gz'
google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Modelin corpus üzerinde eğitilmesi
google_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
google_model.build_vocab(corpus)
google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)


def vectors(x):
    # Film açıklamalarını vektörlere çevirme ve vektörleri saklamak için bir liste oluşturma
    global word_embeddings
    word_embeddings = []

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


# En benzer 5 filmi önerme
def recommendations(movie):
    vectors(df)

    # Kosinüs benzerliğini bulma
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    # Başlık ve Film Görüntüsü linkini alma ve `movies` adı verilen DataFramei oluşturma
    movies = df[['Movie', 'ImgLink']]

    # İndexin ters işlenmesi (Reverse Mapping)
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

# Kullanıcılardan film adı alma ve tavsiyeleri gösterme
while True:
    title = input("Enter title: ")
    print("Recommendations:")
    recommendations(title)

import joblib
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = list(stopwords.words('spanish'))  

def tokenize_text(text):
    return word_tokenize(text)

from DataPreprocessing import TextPreprocessor, TextVectorizer

columns_to_process = ['Titulo', 'Descripcion']
text_preprocessor = TextPreprocessor(columns=columns_to_process)
text_vectorizer_tfidf = TextVectorizer(column='Texto', method='tfidf')

datax = pd.read_csv('./data/fake_news_spanish.csv', sep=';', encoding='utf-8')
data = datax.drop_duplicates()
data = text_preprocessor.transform(data)

X_train, X_test, Y_train, Y_test = train_test_split(
    data[['Texto']], data['Label'], 
    test_size=0.3, stratify=data['Label'], random_state=1
)

X_tfidf_train = text_vectorizer_tfidf.fit_transform(X_train)
X_tfidf_test = text_vectorizer_tfidf.transform(X_test)

arbol = DecisionTreeClassifier(random_state=1)
arbol.fit(X_tfidf_train, Y_train)

pipeline = {
    'preprocessor': text_preprocessor,
    'vectorizer': text_vectorizer_tfidf.vectorizer,
    'model': arbol
}
joblib.dump(pipeline, './assets/pipeline_model.joblib')
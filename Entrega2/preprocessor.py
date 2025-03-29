import joblib
import pandas as pd
import re
import nltk
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Descargar recursos de NLTK si es necesario
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('spanish')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = PorterStemmer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str).apply(self.preprocessor)
        X['Texto'] = X['Titulo'] + ' ' + X['Descripcion']
        return X
    
    def preprocessor(self, text):
        text = re.sub(r'<[^>]*>', '', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        text = text.lower().strip()  
        tokens = self.tokenizer.tokenize(text)  
        filtered_tokens = [self.stemmer.stem(token) for token in tokens if token not in stop_words]
        return ' '.join(filtered_tokens)

class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, column, method='tfidf', ngram_range=(2,3), max_features=100000):
        self.column = column
        self.method = method
        if method == 'bow':
            self.vectorizer = CountVectorizer(tokenizer=word_tokenize, lowercase=True, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features)
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stop_words, lowercase=True, ngram_range=ngram_range, max_features=max_features)
    
    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.column])
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X[self.column])

# Preprocesar y entrenar modelo
data = pd.read_csv('./data/fake_news_spanish.csv', sep=';', encoding='utf-8')
data = data.drop_duplicates()

columns_to_process = ['Titulo', 'Descripcion']
print('iniciando preprocesamiento de texto...')
text_preprocessor = TextPreprocessor(columns=columns_to_process)
print('preprocesamiento de texto finalizado')
print('iniciando vectorizacion...')
text_vectorizer = TextVectorizer(column='Texto', method='tfidf')
print('vectorizacion finalizada')
data = text_preprocessor.transform(data)

X_train, X_test, Y_train, Y_test = train_test_split(data[['Texto']], data['Label'], test_size=0.3, stratify=data['Label'], random_state=1)

X_tfidf_train = text_vectorizer.fit_transform(X_train)
X_tfidf_test = text_vectorizer.transform(X_test)

model = DecisionTreeClassifier(random_state=1)
model.fit(X_tfidf_train, Y_train)

pipeline = {
    'preprocessor': text_preprocessor,
    'vectorizer': text_vectorizer.vectorizer,
    'model': model
}

# Guardar el pipeline correctamente
joblib.dump(pipeline, 'pipeline_model.joblib')


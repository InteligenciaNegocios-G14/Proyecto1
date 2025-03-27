import re
import nltk
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

stop_words = list(stopwords.words('spanish'))

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
    def __init__(self, column, method='bow', ngram_range=(2,3), max_features=100000):
        self.column = column
        self.method = method
        if method == 'bow':
            self.vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=stop_words, lowercase=True, token_pattern=None, ngram_range=ngram_range, max_features=max_features)
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stop_words, lowercase=True, token_pattern=None, ngram_range=ngram_range, max_features=max_features)
    
    def fit(self, X, y=None):
        self.vectorizer.fit(X[self.column])
        return self
    
    def transform(self, X):
        return self.vectorizer.transform(X[self.column])
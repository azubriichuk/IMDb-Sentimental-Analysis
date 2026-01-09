import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#NKLT downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # using lemmatization instead of stemming for better context preservation
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        #limiting data for faster processing during development
        self.df = self.df.iloc[:5000] 
        print(f"Data loaded: {self.df.shape}")

    def clean_text(self, text):
        #lowercase, remove HTML tags and punctuation
        text = text.lower()
        text = re.sub(r'<br\s*/>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # tokenization, stopword removal, lemmatization
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)

    def preprocess_data(self):
        print("Cleaning reviews (removing HTML, punctuation, stopwords, lemmatization)...")
        self.df['cleaned_review'] = self.df['review'].apply(self.clean_text)
        self.df['target'] = self.df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        
        # Feature Engineering: Review Length
        self.df['review_length'] = self.df['cleaned_review'].apply(lambda x: len(x.split()))

    def show_eda(self):
        print("\n--- EDA: Data statistics ---")
        print(self.df['sentiment'].value_counts())
        
        # sentiment distribution
        plt.figure(figsize=(6,4))
        sns.countplot(x='sentiment', data=self.df)
        plt.title('Distribution of Sentiments')
        plt.show()

        # feature analysis: review length
        plt.figure(figsize=(10,6))
        sns.histplot(data=self.df, x='review_length', hue='sentiment', kde=True, bins=50)
        plt.title('Distribution of Review Lengths by Sentiment')
        plt.show()

        # word clouds
        print("Generating WordClouds...")
        pos_text = " ".join(self.df[self.df['sentiment']=='positive']['cleaned_review'])
        neg_text = " ".join(self.df[self.df['sentiment']=='negative']['cleaned_review'])
        
        wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
        wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(wc_pos, interpolation='bilinear')
        axes[0].set_title('Positive Words')
        axes[0].axis('off')
        
        axes[1].imshow(wc_neg, interpolation='bilinear')
        axes[1].set_title('Negative Words')
        axes[1].axis('off')
        plt.show()

    def split_and_vectorize(self):
        print("\nVectorizing text (TF-IDF)...")
        X = self.df['cleaned_review']
        y = self.df['target']

        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.X_train = self.vectorizer.fit_transform(X_train_raw)
        self.X_test = self.vectorizer.transform(X_test_raw)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
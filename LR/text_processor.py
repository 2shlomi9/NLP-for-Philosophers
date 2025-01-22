import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from gensim.models import Word2Vec

class TextProcessor:
    def __init__(self, n=2):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) 
        self.n = n  # This will control the size of the N-grams (e.g., 2 for bigrams)
    
    def clean_text(self, text):
        text = re.sub(r"b[\'\"]", '', text)  
        text = re.sub(r"[^\x00-\x7F]+", '', text)  # Remove non-ASCII characters
        text = re.sub(r"[^a-zA-Z\s]", '', text.lower())  # Remove non-alphabetic characters
        
        tokens = word_tokenize(text)  # Tokenize the text
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]  # Lemmatize and remove stopwords
        
        return tokens

    def read_data(self, filepath):
        
        data = pd.read_csv(filepath)
        sentiment_counts = data['school'].value_counts()

        print(sentiment_counts) # List of schools name and how much have for each school

        schools = ['aristotle', 'german_idealism', 'plato', 'continental']

        for school in schools:
            data[school] = data['school'].apply(lambda x: 1 if x == school else 0)

        train_data, test_data = train_test_split(
        data, 
        test_size=0.2, 
        random_state=42, 
        stratify=data['school']  
        )
        
        y_train = train_data[schools].values
        y_test = test_data[schools].values

        return train_data, test_data, y_train, y_test, schools
    
    def compute_bow(self, documents, data, max_ft, ngram):
        """
        Compute Bag-Of-Bow scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        bow_matrix: Sparse matrix of Bag-Of-Bow scores.
        feature_names: List of feature names (terms).
        """
        if max_ft != -1 and ngram != -1:
            vectorizer = CountVectorizer(max_features=max_ft, ngram_range=(ngram,ngram))
        elif max_ft != -1 and ngram == -1:
            vectorizer = CountVectorizer(max_features=max_ft)
        elif max_ft != -1 and ngram == -1:
            vectorizer = CountVectorizer(ngram_range=(ngram,ngram))
        else:
            vectorizer = CountVectorizer()

        vectorizer.fit(documents)

        bow_matrix = vectorizer.transform(data)
        bow_matrix_array = bow_matrix.toarray()  
        return bow_matrix_array
    
    def compute_tfidf(self, documents, data, max_ft, ngram):
        """
        Compute TF-Idata scores for a list of documents.

        Parameters:
        documents (list of str): List of text documents.

        Returns:
        tfidata_matrix: Sparse matrix of TF-Idata scores.
        feature_names: List of feature names (terms).
        """
        if max_ft != -1 and ngram != -1:
            vectorizer = TfidfVectorizer(max_features=max_ft, ngram_range=(ngram,ngram))
        elif max_ft != -1 and ngram == -1:
            vectorizer = TfidfVectorizer(max_features=max_ft)
        elif max_ft != -1 and ngram == -1:
            vectorizer = TfidfVectorizer(ngram_range=(ngram,ngram))
        else:
            vectorizer = TfidfVectorizer()
            
        vectorizer.fit(documents)

        tfidf_matrix = vectorizer.transform(data)
        tfidf_matrix_array = tfidf_matrix.toarray()
        return tfidf_matrix_array
    
    def convert_to_vector(self, fit_data, transfrom_data, labels, vectorizer = 'bow', max_ft = -1, ngram = -1):
        """
        Process a CSV file to compute TF-Idata matrix for headlines.

        Parameters:
        filepath (str): Path to the CSV file. The file should have a column 'Label' for labels
        and other columns containing text headlines.

        Returns:
        tfidata_matrix: Sparse matrix of TF-Idata scores.
        bow_matrix: Sparse matrix of bow scores.
        labels: List of labels.
        feature_names: List of feature names (terms).
        """ 
        # Extract all texts from the 'sentence_str' column
        combined_texts = fit_data['sentence_str'].astype(str).tolist()

        # Clean the text
        fit_cleaned_texts = [" ".join(self.clean_text(text)) for text in combined_texts]

        # Extract all texts from the 'sentence_str' column
        combined_texts = transfrom_data['sentence_str'].astype(str).tolist()

        # Clean the text
        transform_cleaned_texts = [" ".join(self.clean_text(text)) for text in combined_texts]

        if vectorizer == 'bow':
            data_matrix = self.compute_bow(fit_cleaned_texts, transform_cleaned_texts, max_ft, ngram)
        if vectorizer == 'tfidf':
            data_matrix = self.compute_tfidf(fit_cleaned_texts, transform_cleaned_texts, max_ft, ngram)

        data_matrix = np.hstack((labels, data_matrix))
        print(f"Y shape - {labels.shape}, X shape - {data_matrix.shape}")
        return data_matrix

    def train_word2vec(self, documents, vector_size=100, window=5, min_count=1, epochs=10):
        """
        Train a Word2Vec model on a list of tokenized documents.

        Parameters:
        documents (list of str): List of text documents.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        epochs (int): Number of iterations (epochs) over the corpus.

        Returns:
        model: Trained Word2Vec model.
        """
        # Clean and tokenize documents
        cleaned_documents = [self.clean_text(doc) for doc in documents]
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=cleaned_documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        model.train(cleaned_documents, total_examples=len(cleaned_documents), epochs=epochs)
        print("Word2Vec model trained successfully.")
        return model

    def convert_to_word2vec(self, model, documents):
        """
        Convert documents to sentence embeddings using a trained Word2Vec model.

        Parameters:
        model: Trained Word2Vec model.
        documents (list of str): List of text documents.

        Returns:
        sentence_embeddings: Matrix of sentence embeddings (one vector per document).
        """
        # Clean and tokenize documents
        cleaned_documents = [self.clean_text(doc) for doc in documents]

        # Compute sentence embeddings
        sentence_embeddings = []
        for tokens in cleaned_documents:
            word_vectors = [model.wv[word] for word in tokens if word in model.wv]
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(model.vector_size)  # If no words match, use zero vector
            sentence_embeddings.append(sentence_vector)

        sentence_embeddings = np.array(sentence_embeddings)

        return sentence_embeddings


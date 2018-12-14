from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def VectCount(X_train, X_test):
    vectorizer_count = CountVectorizer(token_pattern = r'\b\w+\b')
    train_matrix_count = vectorizer_count.fit_transform(X_train)
    test_matrix_count = vectorizer_count.transform(X_test)
    features = vectorizer_count.get_feature_names()
    return train_matrix_count, test_matrix_count, features

def VectTfidf(X_train, X_test):
    vectorizer_tfidf = TfidfVectorizer(token_pattern = r'\b\w+\b', norm=None, use_idf=True)
    train_matrix_tfidf = vectorizer_tfidf.fit_transform(X_train)
    test_matrix_tfidf = vectorizer_tfidf.transform(X_test)
    features = vectorizer_tfidf.get_feature_names()
    return train_matrix_tfidf, test_matrix_tfidf, features

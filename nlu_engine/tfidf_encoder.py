from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TfidfEncoder:
    @staticmethod
    def encode_vectors(data, tfidf_vectorizer):
        """
        Create a tfidf vectorizer.
        :param data: pandas dataframe (for training a model), utterance list (for running inference of trained model)
        :return: tfidf vectorized utterances
        """

        if isinstance(data, pd.DataFrame):
            tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
                data)

        elif isinstance(data, str):
            tfidf_utterance_vectors = tfidf_vectorizer.transform([data])

        return tfidf_utterance_vectors


    @staticmethod
    def encode_training_vectors(data_df):
        """
        Create a tfidf vectorizer.
        :param data_df: pandas dataframe (for training a model)
        :return: tfidf vectorized utterances
        """

        tfidf_vectorizer = TfidfVectorizer()

        tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
            data_df['answer_normalised'].values)
        #NOTE: I think it is wrong to return this object, we want to export the fit_transform and use it for the transform!
        return tfidf_utterance_vectors, tfidf_vectorizer

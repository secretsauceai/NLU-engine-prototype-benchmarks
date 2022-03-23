from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TfidfEncoder:
    @staticmethod
    def encode_vectors(data):
        """
        Create a tfidf vectorizer.
        :param data: pandas dataframe (for training a model), utterance list (for running inference of trained model)
        :return: tfidf vectorized utterances
        """

        #NOTE: I think I need to save the tfidf_vectorizer object, so that I can use it to encode the utterances!
        tfidf_vectorizer = TfidfVectorizer()


        # TODO: change from answer_normalised to answer_annotated
        # TODO: write a function to process the answer_annotated

        # TODO: if data_df is a pandas dataframe, then use the data_df.text.values
        if isinstance(data, pd.DataFrame):
            data_df = data
            tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
                data_df['answer_normalised'].values)

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

        #NOTE: I think I need to save the tfidf_vectorizer object, so that I can use it to encode the utterances!
        tfidf_vectorizer = TfidfVectorizer()

        tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
            data_df['answer_normalised'].values)
        #NOTE: I think it is wrong to return this object, we want to export the fit_transform and use it for the transform!
        return tfidf_utterance_vectors, tfidf_vectorizer

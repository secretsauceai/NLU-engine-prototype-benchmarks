from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEncoder:
    @staticmethod
    def encode_vectors(data_df):
        """
        Create a tfidf vectorizer.
        :param data_df: pandas dataframe
        :return: tfidf vectorizer
        """
        tfidf_vectorizer = TfidfVectorizer()
        # TODO: change from answer_normalised to answer_annotated
        # TODO: write a function to process the answer_annotated
        tfidf_utterance_vectors = tfidf_vectorizer.fit_transform(
            data_df['answer_normalised'].values)
        return tfidf_utterance_vectors
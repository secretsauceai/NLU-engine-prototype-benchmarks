from .label_encoder import LabelEncoder
from .tfidf_encoder import TfidfEncoder

class IntentMatcher:
    @staticmethod
    def predict_label(classifier_model, tfidf_vectorizer, utterance):
        #TODO: move this to the intent classifier class
        utterance = utterance.lower()
        print(f'Predicting label for utterance: {utterance}')
        transformed_utterance = TfidfEncoder.encode_vectors(
            utterance, tfidf_vectorizer)
        transformed_utterance = NLUEngine.get_dense_array(
            classifier_model, transformed_utterance)

        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.inverse_transform(predicted_label)
        return decoded_label[0]

    @staticmethod
    def get_incorrect_predicted_labels():
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """
        #TODO: move this to the intent classifier class
        pass

    @staticmethod
    def encode_labels_and_utterances(
        data_df_path,
        labels_to_predict,
        classifier
    ):
        """
        Encode the labels and the utterances.
        :param data_df: pandas dataframe
        :return: intent classifier
        """
        #TODO: move this to the intent classifier class
        data_df = NLUEngine.load_data(data_df_path)

        if labels_to_predict == 'intent':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.intent.values)
        elif labels_to_predict == 'scenario' or labels_to_predict == 'domain':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.scenario.values)

        vectorized_utterances, tfidf_vectorizer = TfidfEncoder.encode_training_vectors(
            data_df)
        vectorized_utterances = NLUEngine.get_dense_array(
            classifier, vectorized_utterances)
        return encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer

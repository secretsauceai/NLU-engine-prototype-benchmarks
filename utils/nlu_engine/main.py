from .label_encoder import LabelEncoder
from .analytics import Analytics
from .intent_matcher import IntentMatcher
from .entity_extractor import EntityExtractor, crf
from .data_utils import DataUtils

class NLUEngine:
    """
    The NLUEngine class is the main class of the NLU engine, made up of intent (domain) labeling and entity extraction.
    It contains all the necessary methods to train, test and evaluate the models.
    """

    @staticmethod
    def train_intent_classifier(
        data_df_path,
        labels_to_predict,
        classifier
        ):
        """
        Train the intent classifier.
        :param data_df: pandas dataframe
        :return: intent classifier model
        """
        #NOTE: This method will stay in main and won't be moved to a separate class
        encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer = IntentMatcher.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
        )
        return IntentMatcher.train_classifier(
            classifier,
            vectorized_utterances,
            encoded_labels_to_predict
            ), tfidf_vectorizer

    @staticmethod
    def evaluate_intent_classifier(
        data_df_path,
        labels_to_predict,
        classifier
    ):
        """
        Evaluates a classifier and generates a report
        """
        print(f'Evaluating {classifier}')
        encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer = IntentMatcher.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
            )

        
        predictions = Analytics.cross_validate_classifier(
            classifier,
            x_train=vectorized_utterances,
            y_train=encoded_labels_to_predict
        )

        predictions_decoded = LabelEncoder.decode(predictions).tolist()
        decoded_labels_to_predict = LabelEncoder.decode(
            encoded_labels_to_predict).tolist()
        report = Analytics.generate_report(
            classifier=classifier,
            predictions_decoded=predictions_decoded,
            decoded_labels_to_predict=decoded_labels_to_predict
        )

        report_df = Analytics.convert_report_to_df(
            classifier=classifier,
            report=report,
            label=labels_to_predict,
            encoding='tfidf'
        )
        return report_df

    @staticmethod
    def train_entity_classifier(data_df):
        """
        Train the entity classifier.
        :param data_df: pandas dataframe
        :return: entity classifier model
        """
        #TODO: check if data_df is a path or a dataframe, if path then load the dataframe
        print('Training entity classifier')
        X, y = EntityExtractor.get_targets_and_labels(data_df)
        crf_model = EntityExtractor.train_crf_model(X, y)
        return crf_model

    @staticmethod
    def create_entity_tagged_utterance(utterance, crf_model):
        """
        runs EntityExtractor.tag_utterance and gets all entity types and entities, formats the utterance into the entity annotated utterance
        This thing is a monster, it really should be refactored, perhaps it can be combined with other functions here to streamline the code
        """
        tagged_utterance = EntityExtractor.tag_utterance(
            utterance,
            crf_model
            )
        split_tagged_utterance = tagged_utterance.split(' ')

        for idx, token in enumerate(split_tagged_utterance):
            if '[' in token:
                if split_tagged_utterance[idx + 5]:
                    if token in split_tagged_utterance[idx + 3]:
                        split_tagged_utterance[idx + 2] = split_tagged_utterance[idx + 2].replace(
                            ']', '') + ' ' + split_tagged_utterance[idx + 5]

                        split_tagged_utterance[idx + 3] = ''
                        split_tagged_utterance[idx + 4] = ''
                        split_tagged_utterance[idx + 5] = ''

        normalised_split_tagged_utterance = [
            x for x in split_tagged_utterance if x]

        formatted_tagged_utterance = ' '.join(
            normalised_split_tagged_utterance)

        return formatted_tagged_utterance


    @staticmethod
    def evaluate_entity_classifier(data_df):
        """
        Evaluates the entity classifier and generates a report
        """

        print('Evaluating entity classifier')

        X, y = EntityExtractor.get_targets_and_labels(data_df)
        predictions = Analytics.cross_validate_classifier(crf, X, y)
        report_df = Analytics.generate_entity_classification_report(
            predictions, y)
        return report_df
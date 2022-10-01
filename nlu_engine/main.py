from .label_encoder import LabelEncoder
from .analytics import Analytics
from .intent_matcher import IntentMatcher
from .entity_extractor import EntityExtractor, crf
from .data_utils import DataUtils
import gc

import pandas as pd


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

        def combine_and_remove_entities(split_tagged_utterance):
            """
            Combines neighboring entities of the same type and marks the duplicates to be removed.
            NOTE: It is important for parsing to keep the same length, therefore we mark them instead of directly remove the matches.
            """
            change_counter = 0
            for index, token in enumerate(split_tagged_utterance):
                if '[' in token:
                    if len(split_tagged_utterance) > index + 3:
                        if token == split_tagged_utterance[index + 3]:
                            split_tagged_utterance[index + 2] = split_tagged_utterance[index + 2].replace(
                                ']', '') + ' ' + split_tagged_utterance[index + 5]

                            split_tagged_utterance[index + 3] = 'to_remove'
                            split_tagged_utterance[index + 4] = 'to_remove'
                            split_tagged_utterance[index + 5] = 'to_remove'
                            change_counter += 1
            return (split_tagged_utterance, change_counter)

        def remove_entities(split_tagged_utterance):
            try:
                while True:
                    split_tagged_utterance.remove('to_remove')
            except ValueError:
                pass
            return split_tagged_utterance

        combined_entities_split_tagged_utterance, change_counter = combine_and_remove_entities(
            split_tagged_utterance)
        removed_entities_split_tagged_utterance = remove_entities(
            combined_entities_split_tagged_utterance)

        while change_counter > 0:
            combined_entities_split_tagged_utterance, change_counter = combine_and_remove_entities(
                removed_entities_split_tagged_utterance)
            removed_entities_split_tagged_utterance = remove_entities(
                combined_entities_split_tagged_utterance)

        return ' '.join(removed_entities_split_tagged_utterance)

    @staticmethod
    def evaluate_entity_classifier(data_df, cv=None):
        """
        Evaluates the entity classifier and generates a report
        """

        print('Evaluating entity classifier')

        X, y = EntityExtractor.get_targets_and_labels(data_df)

        predictions = Analytics.cross_validate_classifier(crf, X, y, cv)
        report_df = Analytics.generate_entity_classification_report(
            predictions, y)
        del X, y, predictions
        gc.collect()
        return report_df

    @staticmethod
    def get_entity_reports_for_domains(data_df):
        """
        Gets the entity reports for all domains in the dataframe, where each domain is evaluated separately.
        """
        domains = data_df['scenario'].unique().tolist()
        domain_entity_reports_df = pd.DataFrame()

        def check_percent_of_non_entities(domain_df):
            number_of_nan_entities = domain_df['entity_types'].isnull().sum()
            total_size = len(domain_df)
            percent_no_entities = number_of_nan_entities / total_size
            return percent_no_entities

        for domain in domains:
            print(f'Evaluating entity classifier for {domain}')
            domain_df = data_df[data_df['scenario'] == domain]

            percent_of_non_entities = check_percent_of_non_entities(domain_df)

            if percent_of_non_entities > 0.90:
                print(
                    f'{domain} has {percent_of_non_entities}% non-entities. It is too sparse to evaluate.'
                )
                continue

            domain_entity_report_df = NLUEngine.evaluate_entity_classifier(
                data_df=domain_df, cv=3)
            domain_scores_df = domain_entity_report_df.tail(3)
            domain_scores_df['domain'] = domain
            domain_entity_reports_df = domain_entity_reports_df.append(
                domain_scores_df)

        return domain_entity_reports_df

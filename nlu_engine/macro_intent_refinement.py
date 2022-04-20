import pandas as pd

from .label_encoder import LabelEncoder

class MacroIntentRefinement:
    @staticmethod
    def rename_overlapping_intents(nlu_data_df, nlu_data_info_df):
        """
        Rename the intents that are overlapping over several domains:
        e.g. 'intent' and 'intent' in domain 'domain_1' and 'domain_2' overlap, therefore they are renamed to 'intent_domain_1' and 'intent_domain_2'
        :param nlu_data_df: pandas dataframe
        :return: pandas dataframe
        """
        overlapping_intents_df = nlu_data_df[nlu_data_df['intent'].isin(
            nlu_data_info_df.index)]

        overlapping_intents_df['intent'] = overlapping_intents_df['scenario'] + \
            '_' + overlapping_intents_df['intent']
        nlu_data_df['intent'].update(overlapping_intents_df['intent'])
        return nlu_data_df

    @staticmethod
    def intent_keyword_feature_rankings(intent_classifier_model, tfidf_vectorizer):
        """
        This function ranks the intents based on their keywords.
        :param intent_classifier_model:
        :param tfidf_vectorizer:
        :return: pandas dataframe
        """
        coefs = intent_classifier_model.coef_
        classes = intent_classifier_model.classes_
        feature_names = tfidf_vectorizer.get_feature_names()

        output = []
        for class_index, features in enumerate(coefs):
            for feature_index, feature in enumerate(features):
                output.append(
                    (classes[class_index], feature_names[feature_index], feature))
        feature_rank_df = pd.DataFrame(
            output, columns=['class', 'feature', 'coef'])

        feature_rank_df['class'] = LabelEncoder.inverse_transform(
            feature_rank_df['class'])

        feature_rank_df["abs_value"] = feature_rank_df["coef"].apply(
            lambda x: abs(x))

        feature_rank_df["colors"] = feature_rank_df["coef"].apply(
            lambda x: "green" if x > 0 else "red")

        return feature_rank_df

    @staticmethod
    def get_incorrect_intent_and_prediction_counts(nlu_data_df, incorrect_intent_predictions_df):
        """
        Get the number of incorrect intent predictions.
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: pandas dataframe
        """

        #TODO: unless these things are needed for the venn wordcloud diagrams, this can be removed

        # get the list of the correct intents for the incorrectly predicted intents
        correct_intents = incorrect_intent_predictions_df['intent'].unique()

        incorrect_intent_prediction_counts = incorrect_intent_predictions_df['predicted_label'].value_counts(
        )

        # get predicted_label to list and rename it to incorrect_predicted_labels
        incorrect_predicted_intents = incorrect_intent_predictions_df['predicted_label'].unique(
        )

        # get count of each intent in incorrect_predicted_intents from the nlu_data_df
        all_intent_counts = nlu_data_df['intent'].value_counts()

        correct_intent_counts = all_intent_counts[all_intent_counts.index.isin(
            incorrect_predicted_intents)]

        # combine columns of correct_intent_counts and incorrect_intent_predictions_count by index
        #TODO: add in column for scenario ?
        domain_intent_counts_df = pd.concat(
            [correct_intent_counts, incorrect_intent_prediction_counts], axis=1)
        domain_intent_counts_df.columns = [
            'correct_total_count', 'incorrect_in_domain_count']
        print(
            f'This is a table of all the incorrectly predicted intents from {incorrect_intent_predictions_df.shape[0]} utterances.\nIt\'s likely that some of the incorrectly predicted intents are outside of the domain you are checking.\nThe correct_total_count is the total number of correct utterances with that intent.\nThe incorrect_in_domain_count is the number of utterances with that intent that are incorrectly predicted in the domain you are checking.\n')

        return domain_intent_counts_df.sort_values(by='incorrect_in_domain_count', ascending=False)

    @staticmethod
    def get_utterance_features(utterance, intent, intent_feature_rank_df):
        """
        Get the features of an utterance.
        :param utterance: string
        :param intent_feature_rank_df: pandas dataframe
        :return: list of tuples
        """
        features = []
        for word in utterance.split(' '):
            try:
                coef = intent_feature_rank_df[(intent_feature_rank_df['class'] == intent) & (
                    intent_feature_rank_df['feature'] == word)]['coef'].values[0]
            except:
                coef = 0
            features.append((word, coef))
        return features

    @staticmethod
    def get_incorrect_predicted_intents_report(nlu_domain_df, incorrect_intent_predictions_df, intent_report_df, intent_feature_rank_df):
        """
        Get a report of the incorrectly predicted intents
        :param nlu_domain_df: pandas dataframe
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: pandas dataframe
        """
        # get an array of the correct intents for the incorrectly predicted intents
        intent_values = incorrect_intent_predictions_df['intent'].unique()

        # json with intent, f1 score, total count, total incorrect count, notes, incorrect predicted intents and their counts
        incorrect_predicted_intents_report = {}

        # for every intent in intent_column_values, get the value_counts of the intent and a list of the predicted intents and their values
        for intent in intent_values:
            f1_for_intent = intent_report_df[intent_report_df['intent'].str.contains(
                intent)]['f1-score'].round(2).values[0]
            all_examples_of_intent_df = nlu_domain_df[(
                nlu_domain_df["intent"] == intent)]

            correct_utterance_annotated_example = all_examples_of_intent_df[~all_examples_of_intent_df.index.isin(
                incorrect_intent_predictions_df.index)]['answer_annotation'].iloc[0]
            correct_utterance_example = all_examples_of_intent_df[~all_examples_of_intent_df.index.isin(
                incorrect_intent_predictions_df.index)]['answer_normalised'].iloc[0]

            incorrect_utterance_annotated_example = incorrect_intent_predictions_df[
                incorrect_intent_predictions_df["intent"] == intent]["answer_annotation"].iloc[0]

            incorrect_utterance_example = incorrect_intent_predictions_df[
                incorrect_intent_predictions_df["intent"] == intent]["answer_normalised"].iloc[0]

            incorrect_utterance_intent = incorrect_intent_predictions_df[
                incorrect_intent_predictions_df["intent"] == intent]["predicted_label"].iloc[0]

            top_features = intent_feature_rank_df[intent_feature_rank_df['class'] == intent].sort_values(
                'coef', ascending=False)['feature'].head(10).to_list()

            # Get overlapping features from top_features with the rest of the intent_feature_rank_df, return the features and the intents they overlap with
            #TODO: group by intent or feature?
            top_10_features_per_intent_df = intent_feature_rank_df.sort_values(
                'coef', ascending=False).groupby('class').head(10)
            overlapping_features = top_10_features_per_intent_df[(top_10_features_per_intent_df['class'] != intent) & (
                top_10_features_per_intent_df['feature'].isin(top_features))].drop(['colors', 'abs_value'], axis=1).rename(columns={'class': 'intent'}).to_dict(orient='records')


            incorrect_predicted_intents_report[intent] = {
                'f1_score': f1_for_intent,
                'total_count': all_examples_of_intent_df.shape[0],
                'total_incorrect_count': incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].shape[0],
                'top_features': top_features,
                'overlapping_features': overlapping_features,
                'correct_utterance_example': [
                    intent,
                    correct_utterance_annotated_example,
                    MacroIntentRefinement.get_utterance_features(
                        correct_utterance_example, intent, intent_feature_rank_df)
                ],
                'incorrect_utterance_example': [
                    incorrect_utterance_intent,
                    incorrect_utterance_annotated_example,
                    MacroIntentRefinement.get_utterance_features(
                        incorrect_utterance_example, intent, intent_feature_rank_df)
                ],
                'incorrect_predicted_intents_and_counts': incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].predicted_label.value_counts().to_dict()
            }
        return incorrect_predicted_intents_report

    @staticmethod
    def get_intent_dataframes_to_refine(incorrect_intent_predictions_df):
        """
        Get a dictionary of the incorrectly predicted dataframes by intent sorted by predicted intent.
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: dictionary of pandas dataframes
        """
        # get an array of the correct intents for the incorrectly predicted intents
        intent_values = incorrect_intent_predictions_df['intent'].unique()
        dataframe_dictionary = {}

        # for every intent in intent_column_values, get the value_counts of the intent and a list of the predicted intents and their values
        for intent in intent_values:
            df = incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].sort_values(
                by='predicted_label')
            dataframe_dictionary[intent] = df

        return dataframe_dictionary

    @staticmethod
    def list_and_select_intent(incorrect_intent_predictions_df):
        """
        List and select the intent to refine.
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: string
        """
        intent_values = incorrect_intent_predictions_df['intent'].unique()
        intent_to_refine = input(f"select intent to refine from the list:\n{intent_values}")
        return intent_to_refine
import pandas as pd
import ipysheet


class MacroDataRefinement:
    """
    This is where the magic NLU refinement happens.
    """

    @staticmethod
    def get_data_info(nlu_data_df):
        number_of_domains = len(nlu_data_df['scenario'].unique())
        list_of_domains = nlu_data_df['scenario'].unique()

        number_of_intents = nlu_data_df['intent'].nunique()
        list_of_intents = nlu_data_df['intent'].unique()

        number_of_utterances = nlu_data_df['answer_normalised'].nunique()

        number_of_intents_per_domain_df = nlu_data_df.pivot_table(
            index='intent', columns='scenario', values='answer_normalised', aggfunc='count')

        overlapping_intents_in_domains_df = number_of_intents_per_domain_df.dropna(
            subset=list_of_domains, thresh=2).dropna(axis=1, how='all')

        print(
            f'From a total of {number_of_utterances} utterances, there are {number_of_domains} domains, and {number_of_intents} intents\n')

        print(f'List of domains: {list_of_domains}\n')

        print(f'List of intents: {list_of_intents}\n')

        if overlapping_intents_in_domains_df.empty:
            print(
                'There are no overlapping intents in the domains (that\'s a good thing)')
            output_df = number_of_intents_per_domain_df
        else:
            print(
                f'Uh oh. There are {overlapping_intents_in_domains_df.shape[0]} intents with overlapping domains.\nThat\'s usually not good to have.\n\nThe easiest solution would be to rename them in respect to their domains.')
            output_df = overlapping_intents_in_domains_df

        return output_df

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
    def get_incorrect_intent_and_prediction_counts(nlu_data_df, incorrect_intent_predictions_df):
        """
        Get the number of incorrect intent predictions.
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: pandas dataframe
        """

        #TODO: unless these things are needed for the venn wordcloud diagrams, this can be removed

        # get the list of the correct intents for the incorrectly predicted intents
        correct_intents = incorrect_intent_predictions_df['intent'].unique()

        #TODO: this method is a work in progress and untested besides in the notebook itself
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
    def get_incorrect_predicted_intents_report(nlu_domain_df, incorrect_intent_predictions_df):
        """
        Get a report of the incorrectly predicted intents
        :param nlu_domain_df: pandas dataframe
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: pandas dataframe
        """
        # get an array of the correct intents for the incorrectly predicted intents
        intent_values = incorrect_intent_predictions_df['intent'].unique()



        # for every intent in intent_column_values, get the value_counts of the intent and a list of the predicted intents and their values
        for intent in intent_values:
            all_examples_of_intent_df = nlu_domain_df[(nlu_domain_df["intent"] == intent)]
            correct_utterance_example = all_examples_of_intent_df[~all_examples_of_intent_df.index.isin(
                incorrect_intent_predictions_df.index)]['answer_annotation'].iloc[0]
            print(
                f'intent: {intent}, total count: {all_examples_of_intent_df.shape[0]}, total incorrect count: {incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].shape[0]}\n example of correctly predicted utterance: {correct_utterance_example}\n example of incorrectly predicted utterance: {incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent]["answer_annotation"].iloc[0]}\n')
            print(
                f'incorrect predicted intents for {intent} and their counts:\n{incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].predicted_label.value_counts().to_string()}\n')

    @staticmethod
    def get_intent_dataframes_to_refine(incorrect_intent_predictions_df):
        """
        Get a dictionary of the incorrectly predicted dataframes by intent sorted by predicted intent.
        :param incorrect_intent_predictions_df: pandas dataframe
        :return: dictionary of pandas dataframes
        """
        # get an array of the correct intents for the incorrectly predicted intents
        intent_values = incorrect_intent_predictions_df['intent'].unique()
        print(f'intents: {intent_values}')
        dataframe_dictionary = {}

        # for every intent in intent_column_values, get the value_counts of the intent and a list of the predicted intents and their values
        for intent in intent_values:
            df = incorrect_intent_predictions_df[incorrect_intent_predictions_df["intent"] == intent].sort_values(by='predicted_label')
            dataframe_dictionary[intent] = df

        return dataframe_dictionary


    @staticmethod
    def create_sheet(to_review_df):
        """
        Create a sheet from a dataframe
        :param df_to_review: pandas dataframe
        :return: IPySheet
        """
        to_review_df.drop(
            columns=['answer_normalised', 'question'], inplace=True)

        to_review_df = to_review_df.assign(review=None)
        to_review_df['review'] = to_review_df['review'].astype(bool)

        to_review_df = to_review_df.assign(move=None)
        to_review_df['move'] = to_review_df['move'].astype(bool)

        to_review_df = to_review_df.assign(remove=None)
        to_review_df['remove'] = to_review_df['remove'].astype(bool)

        to_review_sheet = ipysheet.from_dataframe(to_review_df)

        return to_review_sheet

    @staticmethod
    def convert_sheet_to_dataframe(sheet):
        """
        Convert a sheet to a dataframe
        :param sheet: IPySheet
        :return: pandas dataframe
        """
        df = ipysheet.to_dataframe(sheet)
        df.index = pd.to_numeric(df.index)

        return df

    @staticmethod
    def remove_entries_marked_remove(dataframe):
        print('Removing all entries marked as "remove"')
        dataframe = dataframe[dataframe['remove'] == False]
        return dataframe

    @staticmethod
    def move_entry(row):
        """
        If a row is marked to be moved, get user input for the correct intent
        :param row: pandas dataframe row
        :return: pandas dataframe row
        """
        if row['move'] == True:
            print(f'The utterance: {row.answer_annotation}\nwith intent: {row.intent}\nand predicted intent: {row.predicted_label}\nwas marked to be moved. Which intent should it be moved to?\nIf {row.predicted_label} is correct, just hit enter/return.')
            corrected_intent = input()
            if corrected_intent == '':
                row['intent'] = row['predicted_label']
            else:
                row['intent'] = corrected_intent
            row['move'] = False
        return row


    @staticmethod
    def refine_intent(data_df, classifier, model_path):
        """
        Refine the intent of the utterances.
        :param data_df: pandas dataframe
        :param classifier: classifier
        :param model_path: path to the pickle file
        :return: None
        """
        pass
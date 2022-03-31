import pandas as pd

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
        #TODO: this method is a work in progress and untested besides in the notebook itself
        incorrect_intent_prediction_counts = incorrect_intent_predictions_df['predicted_label'].value_counts(
        )

        # get predicted_label to list and rename it to incorrect_predicted_labels
        incorrect_intents = incorrect_intent_predictions_df['predicted_label'].unique(
        )

        # do the same as above cell with the correct_intents ('intents')
        # TODO: Why is this unused?? 
        correct_intents = incorrect_intent_predictions_df['intent'].unique()

        # get count of each intent in incorrect_intents from the nlu_data_df
        intent_counts = nlu_data_df['intent'].value_counts()

        correct_intent_counts = intent_counts[intent_counts.index.isin(
            incorrect_intents)]


        # combine columns of correct_intent_counts and incorrect_intent_predictions_count by index
        #TODO: add in column for scenario ?
        domain_intent_counts_df = pd.concat(
            [correct_intent_counts, incorrect_intent_prediction_counts], axis=1)
        domain_intent_counts_df.columns = ['correct_count', 'incorrect_count']

        return domain_intent_counts_df

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
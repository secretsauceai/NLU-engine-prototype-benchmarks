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
    def list_and_select_domain(nlu_data_df):
        """
        List the domains and ask the user to select one.
        :param nlu_data_df: pandas dataframe
        :return: string
        """
        list_of_domains = nlu_data_df['scenario'].unique()
        domain_selection = input(
            f'Please select a domain from the list:\n{list_of_domains}')
        return domain_selection

    @staticmethod
    def create_review_df(to_review_df):
        """
        Create a dataframe with the entries to review. There is a difference between intent and entity review dataframes. The dataframe will be used to create a sheet. 
        :param to_review_df: pandas dataframe
        :return: pandas dataframe
        """

        intent_refinement = False
        to_review_df.drop(
            columns=['answer_normalised', 'question'], inplace=True)
        try:
            to_review_df.drop(columns=['entities', 'entity_types'], inplace=True)
        except:
            intent_refinement = True

        to_review_df = to_review_df.assign(review=None)
        to_review_df['review'] = to_review_df['review'].astype(bool)

        if intent_refinement:
            to_review_df = to_review_df.assign(move=None)
            to_review_df['move'] = to_review_df['move'].astype(bool)
        else:
            to_review_df = to_review_df.assign(use_predicted_tagging=None)
            to_review_df['use_predicted_tagging'] = to_review_df['use_predicted_tagging'].astype(bool)

        to_review_df = to_review_df.assign(remove=None)
        to_review_df['remove'] = to_review_df['remove'].astype(bool)

        return to_review_df

    @staticmethod
    def remove_intent(to_review_df, intent_to_remove):
        to_remove_df = to_review_df[to_review_df['intent'] == intent_to_remove]
        to_remove_df = MacroDataRefinement.create_review_df(to_remove_df)
        to_remove_df['remove'] = True
        return to_remove_df

    @staticmethod
    def create_sheet(to_review_df):
        """
        Create a sheet from a dataframe
        :param df_to_review: pandas dataframe
        :return: IPySheet
        """
        to_review_df = MacroDataRefinement.create_review_df(to_review_df)

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
                #TODO: look up intent for the scenario and set row['scenario'] = corrected_scenario!!!
            row['move'] = False
        return row

    @staticmethod
    def replace_annotated_utterance_with_predicted_tagging(row):
        """
        If a row is marked use_predicted_tagging, replace the annotated utterance with the predicted tagging
        :param row: pandas dataframe row
        :return: pandas dataframe row
        """
        if row['use_predicted_tagging'] == True:
            row['answer_annotation'] = row['predicted_tagging']
        return row

    @staticmethod
    def mark_entries_as_refined(refined_dataframe, refined_type):
        """
        Mark all entries as refined by refined type
        :param refined_dataframe: pandas dataframe
        :param refined_type: string (intent or entity)
        :return: pandas dataframe
        """
        refined_type = refined_type + '_refined'
        refined_dataframe[refined_type] = True
        return refined_dataframe



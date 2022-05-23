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
                #TODO: look up intent for the scenario and set row['scenario'] = corrected_scenario!!!
            row['move'] = False
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

    @staticmethod
    def merge_refined_data_into_dataset(dataset_df, refined_dataframe):
        """
        Merge the refined data into the complete dataset and return the updated dataset
        :param dataset_df: pandas dataframe
        :param refined_dataframe: pandas dataframe
        :return: pandas dataframe
        """
        #TODO: 
        #TODO: should I leave it as is with the intent and entity columns or add those in as a parameter?
        #TODO: There must be a nicer way to do this, but I can never remember pandas syntax, LOL!
        combined_df = dataset_df.merge(refined_dataframe, how='left',
                                      left_index=True,
                                      right_index=True)
        combined_df['scenario_y'].fillna(combined_df['scenario_x'], inplace=True)
        combined_df['intent_y'].fillna(combined_df['intent_x'], inplace=True)
        combined_df['answer_annotation_y'].fillna(
            combined_df['answer_annotation_y'], inplace=True)
        combined_df['status_y'].fillna(combined_df['status_x'], inplace=True)
        #NOTE: hard coded intent_refined and entity_refined for now!
        if 'intent_refined_y' not in combined_df.columns:
            combined_df['intent_refined'].fillna(False, inplace=True)
        if 'entity_refined' in combined_df.columns:
            combined_df['entity_refined'].fillna(False, inplace=True)
        else:
            combined_df['entity_refined'] = False
        combined_df.drop(columns=[
            'scenario_x',
            'intent_x',
            'answer_annotation_x',
            'status_x',
            'move'], inplace=True)
        combined_df.rename(columns={
            'scenario_y': 'scenario',
            'intent_y': 'intent',
            'answer_annotation_y': 'answer_annotation',
            'status_y': 'status'}, inplace=True)

        return combined_df

    @staticmethod
    def upgrade_dataframe(data_df):
        updated_df = pd.DataFrame(columns=[
            'userid',
            'answerid',
            'notes',
            'question',
            'suggested_entities',
            'answer',
            'answer_normalised',
            'scenario',
            'intent',
            'predicted_label',
            'intent_refined',
            'entity_refined',
            'remove',
            'status',
        ])
        updated_df = updated_df.append(data_df)
        return updated_df

    @staticmethod
    def update_dataframe(data_df, refined_intent_df):
        """
        Updates the dataframe with the refined data, if a previously updated dataframe doesn't exist, it formats the original dataframe correctly.
        """
        if 'predicted_label' in data_df.columns:
            updated_df = data_df.copy()
            updated_df.update(refined_intent_df)
            print('Successfully updated dataframe')
            return updated_df
        else:
            print("Dataframe hasn't been upgraded, make sure to upgrade the dataframe first.")
            return None



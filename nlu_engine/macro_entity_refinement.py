import pandas as pd
import numpy as np

class MacroEntityRefinement:
    """
    Macro Data Refinement focused on entities.
    """

    @staticmethod
    def get_incorrect_predicted_entities_report(domain_df, entity_report_df, domain_entity_reports_df):
        """
            Get a report of the incorrectly predicted entities
            :param nlu_domain_df: pandas dataframe
            :param incorrect_intent_predictions_df: pandas dataframe
            :return: pandas dataframe
            """

        incorrect_predicted_entities_report = {}
        
        domain_selection = domain_df['scenario'].unique().tolist()[0]
        entity_types_df = domain_df.dropna(subset=['entity_types'])
        entries = entity_types_df['entity_types'].apply(
            lambda x: x.split(',')).to_list()

        domain_entity_types = set(sum(entries, []))

        domain_f1_score = domain_entity_reports_df[
            domain_entity_reports_df['domain'] == domain_selection
        ]['f1-score'].values[0]

        for entity_type in domain_entity_types:
            entity_entries_df = entity_types_df[
                entity_types_df['entity_types'].str.contains(entity_type)
            ]
            correct_entries_df = entity_entries_df[
                entity_entries_df['answer_annotation'] == entity_entries_df['predicted_tagging']
            ]

            incorrect_entries_df = entity_entries_df[
                entity_entries_df['answer_annotation'] != entity_entries_df['predicted_tagging']
            ]

            incorrect_entities_count = len(incorrect_entries_df.index)
            correct_entities_count = len(correct_entries_df.index)
            total_entities_count = len(entity_entries_df.index)
            print(f'Entity type: {entity_type}')
            print(f'Correct entities: {correct_entities_count}')
            print(f'Incorrect entities: {incorrect_entities_count}')
            print(f'Total entities: {total_entities_count}')

            try:
                correct_utterance_example = correct_entries_df['answer_annotation'].iloc[0]
            except:
                correct_utterance_example = 'There are no correct utterances!'

            try:
                incorrect_utterance_example = incorrect_entries_df['answer_annotation'].iloc[0]
            except:
                incorrect_utterance_example = 'There are no incorrect utterances!'

            f1_score = entity_report_df[
                entity_report_df['entity-type'] == entity_type
            ]['f1-score'].values[0]

            incorrect_predicted_entities_report[entity_type] = {
                'f1-score': f1_score,
                'total_count': total_entities_count,
                'total_incorrect_count': incorrect_entities_count,
                'correct utterance': correct_utterance_example,
                'incorrect utterance': incorrect_utterance_example
            }
        return incorrect_predicted_entities_report

    @staticmethod
    def get_overlapping_entity_types_and_words(df):
        """
            Get the overlapping entity types and words.
            :param df: pandas dataframe
            :return: pandas dataframe
            """
        entity_types = []
        entity_words = []

        def get_entity_types_and_words(row):
            if row['entities'] is not np.nan:
                for entity in row['entities']:
                    entity_types.append(entity['type'])
                    entity_words.append(" ".join(entity['words']))

        df.apply(get_entity_types_and_words, axis=1)
        entity_types_and_words_df = pd.DataFrame(
            {'entity_type': entity_types, 'entity_words': entity_words})
        entity_types_and_words_df.reset_index(inplace=True)

        def count_and_group_combinations(entity_types_and_words_df):
            all_entity_words = entity_types_and_words_df["entity_words"]

            entity_type_word_counts_df = entity_types_and_words_df[all_entity_words.isin(all_entity_words[all_entity_words.duplicated(
            )])].sort_values(by=['entity_words']).groupby(['entity_words', 'entity_type']).count()
            return entity_type_word_counts_df

        entity_type_word_counts_df = count_and_group_combinations(
            entity_types_and_words_df)

        overlapping_entities_df = pd.DataFrame(
            entity_type_word_counts_df).unstack().dropna(thresh=2)

        return overlapping_entities_df

    @staticmethod
    def pick_correct_entity_type_for_overlapping_entities(df):
        """
                Pick the correct entity type for overlapping entities.
                :param df: pandas dataframe
                :return: pandas dataframe
                """
        overlapping_entities_df = MacroEntityRefinement.get_overlapping_entity_types_and_words(
            df)
        to_refine_df = overlapping_entities_df.stack().reset_index()

        overlapping_entity_words = overlapping_entities_df.index.values

        def get_correct_entity_types(overlapping_entity_words, to_refine_df):
            correct_entity_types = []
            for entity_word in overlapping_entity_words:
                entity_types = to_refine_df[to_refine_df["entity_words"]
                                            == entity_word]["entity_type"].values

                correct_entity_type = input(
                    f"Type in the correct entity type for these words\nentity words: {entity_word}\nentity types: {entity_types}"
                    )
                correct_entity_types.append(correct_entity_type)
            return correct_entity_types

        correct_entity_types = get_correct_entity_types(
            overlapping_entity_words, to_refine_df)
        incorrect_entity_types = []
        for entity_type in to_refine_df['entity_type'].values:
            if entity_type not in correct_entity_types:
                incorrect_entity_types.append(entity_type)

        return correct_entity_types, incorrect_entity_types, overlapping_entity_words



    @staticmethod
    def remove_entity(df, entity_to_remove):
        """
            Remove all entries of an entity type from the dataframe.
            :param df: pandas dataframe
            :return: pandas dataframe
            """
        updated_df = df.copy()
        updated_df.loc[updated_df['answer_annotation'].str.contains(
            entity_to_remove), 'remove'] = True
        return updated_df

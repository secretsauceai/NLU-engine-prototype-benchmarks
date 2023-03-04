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
    def get_sorted_incorrect_entity_types_and_counts(incorrect_entity_types, incorrect_entity_counts):
        """
            Get sorted incorrect entity types and counts.
            :param incorrect_entity_types: list
            :param incorrect_entity_counts: list
            :return: series of tuples
            """
        entity_types_and_counts = zip(
            incorrect_entity_types, incorrect_entity_counts)
        
        # arrange the entities in entity_types by total count in ascending order
        entity_types_and_counts = sorted(entity_types_and_counts, key=lambda x: x[1])

        # create a list of the entities in entity_types in ascending order of total count
        entity_types = [entity_type for entity_type, count in entity_types_and_counts]

        # create a list of the total counts of the entities in entity_types in ascending order of total count
        entity_counts = [count for entity_type, count in entity_types_and_counts]

        sorted_entity_types = [entity_type for entity_type, count in sorted(
            zip(entity_types, entity_counts), key=lambda x: x[1])]
        
        sorted_entity_counts = [count for entity_type, count in sorted(
            zip(entity_types, entity_counts), key=lambda x: x[1])]
        
        sorted_entity_types_and_counts = zip(
            sorted_entity_types, sorted_entity_counts)

        #TODO: refactor above code to be more compact

        return sorted_entity_types_and_counts
    
    @staticmethod
    def get_entries_with_sparse_total(domain_df, incorrect_entity_types, incorrect_entity_counts):
        """
            Get the entries with sparse total.
            :param domain_df: pandas dataframe
            :param incorrect_entity_types: list
            :param incorrect_entity_counts: list
            :return: pandas dataframe
        """
        entity_types_with_sparse_total = []
        entity_counts_with_sparse_total = []

        sorted_entity_types_and_counts = MacroEntityRefinement.get_sorted_incorrect_entity_types_and_counts(
            incorrect_entity_types, incorrect_entity_counts)
        
        for entity_type, entity_count in sorted_entity_types_and_counts:
            if entity_count < 5:
                print(f'Entity type: {entity_type}, Total count: {entity_count}')
                entity_types_with_sparse_total.append(entity_type)
                entity_counts_with_sparse_total.append(entity_count)
        if len(entity_types_with_sparse_total) > 0:
            sparse_total_df = pd.DataFrame(
                {
                    'entity_type': entity_types_with_sparse_total,
                    'total_count': entity_counts_with_sparse_total
                })
            sparse_entity_entries_df = domain_df[domain_df['entity_types'].str.contains(
                ('|').join(entity_types_with_sparse_total), na=False)]
            return sparse_total_df, sparse_entity_entries_df
        else:
            return None


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
                    f"Type in the correct entity type for these words \n entity words: {entity_word} \n entity types: {entity_types} \n Press enter to skip this one"
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
    def refine_overlapping_entity_types(df, correct_entity_types, incorrect_entity_types, overlapping_entity_words):
        """
                Refine the entity types.
                :param df: pandas dataframe
                :param correct_entity_types: list
                :param incorrect_entity_types: list
                :param overlapping_entity_words: list
                :return: pandas dataframe
                """
        refined_df = df.copy()

        for incorrect_entity_type, correct_entity_type, overlapping_entity_word in zip(incorrect_entity_types, correct_entity_types, overlapping_entity_words):
            if correct_entity_type != '':
                pattern = f'{incorrect_entity_type} : {overlapping_entity_word}'
                replacement = f'{correct_entity_type} : {overlapping_entity_word}'
                print(f'replacing: {pattern}\nwith: {replacement}')
            
                refined_df['answer_annotation'] = refined_df.answer_annotation.str.replace(
                    pattern, replacement)
        return refined_df

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

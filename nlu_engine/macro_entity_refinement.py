import pandas as pd

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

        print(f'Domain: {domain_selection}\nf1-score: {domain_f1_score}')

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

            print(
                f'entity type: {entity_type}\nf1-score: {f1_score}\n correct utterance: {correct_utterance_example}'
            )


            incorrect_predicted_entities_report[entity_type] = {
                'f1-score': f1_score,
                'total_count': total_entities_count,
                'total_incorrect_count': incorrect_entities_count,
                'correct utterance': correct_utterance_example,
                'incorrect utterance': incorrect_utterance_example
            }
        return incorrect_predicted_entities_report


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

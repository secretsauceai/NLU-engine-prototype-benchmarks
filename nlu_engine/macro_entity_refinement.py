import pandas as pd

class MacroEntityRefinement:
    """
    Macro Data Refinement focused on entities.
    """

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

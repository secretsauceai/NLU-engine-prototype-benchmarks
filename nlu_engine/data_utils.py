from operator import index
import pandas as pd
import pickle
from .entity_extractor import EntityExtractor
from sklearn.naive_bayes import GaussianNB
import json

NB = GaussianNB()

class DataUtils:
    """
    This class handles basic data functions like loading, converting, saving, etc.
    """

    @staticmethod
    def load_data(data):
        """
        Load data from a csv file or import from a pandas dataframe.
        :param data: path to the csv file or the pandas dataframe
        :return: pandas dataframe
        """

        if isinstance(data, str):
            data_df = pd.read_csv(data, sep=';')
            if {'Unnamed: 0'}.issubset(data_df):
                data_df.index = data_df['Unnamed: 0']
                data_df.drop('Unnamed: 0', axis=1, inplace=True)
                data_df.index.name = None
        elif isinstance(data, pd.DataFrame):
            data_df = data
        return data_df.dropna(axis=0, how='any', subset=['answer_annotation', 'scenario'])
    

    @staticmethod
    def convert_annotated_utterances_to_normalised_utterances(data):
        """
        Convert the annotated utterances to normalized utterances.
        :param data: annotated utterance string or pandas dataframe
        :return: string or pandas dataframe
        """

        if isinstance(data, str):
            normalised_data = EntityExtractor.normalise_utterance(utterance=data)

        elif isinstance(data, pd.DataFrame):
            data_df = data
            data_df['answer_normalised'] = data_df['answer_annotation'].apply(
                EntityExtractor.normalise_utterance)
            normalised_data = data_df
        return normalised_data

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
            print(
                "Dataframe hasn't been upgraded, make sure to upgrade the dataframe first.")
            return None

    @staticmethod
    def prepare_dataframe_for_refinement(data_df):
        """
        Prepares the dataframe for refinement.
        """
        #TODO: Will need to refactor for entity refinement
        prepared_data_df = data_df.drop(
                columns=[
                    'userid', 'notes', 'answer', 'answerid', 'suggested_entities', 'intent_refined', 'remove', 'status', 'entity_refined'
                ])
        print('Successfully loaded dataframe')
        return prepared_data_df

    @staticmethod
    def get_domain_df(data_df, domain_selection):
        """
        Gets the domain dataframe.
        """
        prepared_data_df = DataUtils.prepare_dataframe_for_refinement(data_df)
        domain_df = prepared_data_df[prepared_data_df['scenario'] == domain_selection]
        return domain_df



    @staticmethod
    def pickle_model(classifier, model_path):
        """
        Export the model to a pickle file.
        :param classifier: classifier
        :param model_path: path to the pickle file
        :return: None
        """

        with open(model_path, 'wb') as file:
            pickle.dump(classifier, file)

    @staticmethod
    def import_pickled_model(model_path):
        """
        Import pickled model.
        :param model_path: path to the pickle file
        :return: binary of model
        """
        loaded_model = pickle.load(open(model_path, 'rb'))
        return loaded_model

    @staticmethod
    def export_onnx_model(classifier, model_path):
        """
        Export the model to an onnx file like this: http://onnx.ai/sklearn-onnx/
        :param classifier: classifier
        :param model_path: path to the onnx file
        :return: None
        """
        pass


    @staticmethod
    def save_json(data, path):
        """
        Save data to a json file.
        :param data: data to save
        :param path: path to the json file
        :return: None
        """

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_json(path):
        """
        Load data from a json file.
        :param path: path to the json file
        :return: data
        """

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

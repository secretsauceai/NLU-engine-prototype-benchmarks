import pandas as pd
import pickle
from .entity_extractor import EntityExtractor
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

class DataUtils:
    @staticmethod
    def load_data(data):
        """
        Load data from a csv file or import from a pandas dataframe.
        :param data: path to the csv file or the pandas dataframe
        :return: pandas dataframe
        """
        #TODO: move to data_utils.py
        if isinstance(data, str):
            data_df = pd.read_csv(data, sep=';')
        elif isinstance(data, pd.DataFrame):
            data_df = data
        return data_df.dropna(axis=0, how='any', subset=['answer_normalised', 'scenario'])
    

    @staticmethod
    def convert_annotated_utterances_to_normalised_utterances(data):
        """
        Convert the annotated utterances to normalized utterances.
        :param data: annotated utterance string or pandas dataframe
        :return: string or pandas dataframe
        """
        #TODO: move to data_utils.py

        if isinstance(data, str):
            normalised_data = EntityExtractor.normalise_utterance(utterance=data)

        elif isinstance(data, pd.DataFrame):
            data_df = data
            data_df['answer_normalised'] = data_df['answer_annotation'].apply(
                EntityExtractor.normalise_utterance)
            normalised_data = data_df
        return normalised_data     

    @staticmethod
    def get_dense_array(classifier, x_train):
        """
        When using NB classifier, convert the utterances to a dense array.
        :param x_train: tfidf numpy array
        :return: tfidf dense numpy array
        """
        #TODO: move to data_utils.py, use NB in data_utils.py??

        if classifier is NB:
            x_train = x_train.todense()
        else:
            pass
        return x_train

    @staticmethod
    def pickle_model(classifier, model_path):
        """
        Export the model to a pickle file.
        :param classifier: classifier
        :param model_path: path to the pickle file
        :return: None
        """
        #TODO: move to data_utils.py

        with open(model_path, 'wb') as file:
            pickle.dump(classifier, file)

    @staticmethod
    def export_onnx_model(classifier, model_path):
        """
        Export the model to an onnx file like this: http://onnx.ai/sklearn-onnx/
        :param classifier: classifier
        :param model_path: path to the onnx file
        :return: None
        """
        #TODO: move to data_utils.py
        pass

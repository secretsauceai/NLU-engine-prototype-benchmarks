import pandas as pd
import pickle
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
 


from .label_encoder import LabelEncoder
from .tfidf_encoder import TfidfEncoder
from .analytics import Analytics
from .entity_extractor import EntityExtractor, crf


LR = LogisticRegression(
    solver='liblinear',
    random_state=0
)
DT = DecisionTreeClassifier(random_state=42)
ADA = AdaBoostClassifier(n_estimators=100)
KN = KNeighborsClassifier(n_neighbors=100)
RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=0
)
SVM = svm.SVC(
    gamma='scale'
)
NB = GaussianNB()


class NLUEngine:
    """
    The NLUEngine class is the main class of the NLU engine, made up of intent (domain) labeling and entity extraction.
    It contains all the necessary methods to train, test and evaluate the models.
    """
    #TODO: All the methods related to the intent classifier should be moved to the intent classifier class
    #CHALLENGE: Some of the methods for the intent classifier depend on the NLUEngine class, what is the best way to solve this?
    #SOLUTION: file data_utils.py and move in all modeling and data stuff

    @staticmethod
    def load_data(data):
        """
        Load data from a csv file or import from a pandas dataframe.
        :param data: path to the csv file or the pandas dataframe
        :return: pandas dataframe
        """
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
        if classifier is NB:
            x_train = x_train.todense()
        else:
            pass
        return x_train

    # TODO: refactor the training, predictions, etc. into a separate class specifically for intent

    @staticmethod
    def train_classifier(classifier, x_train, y_train):
        # TODO: add in training time
        print(f'Training {str(classifier)}')
        x_train = NLUEngine.get_dense_array(classifier, x_train)
        return classifier.fit(x_train, y_train)

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
    def export_onnx_model(classifier, model_path):
        """
        Export the model to an onnx file like this: http://onnx.ai/sklearn-onnx/
        :param classifier: classifier
        :param model_path: path to the onnx file
        :return: None
        """
        pass

    @staticmethod
    def predict_label(classifier_model, tfidf_vectorizer, utterance):
        #TODO: move this to the intent classifier class
        utterance = utterance.lower()
        print(f'Predicting label for utterance: {utterance}')
        transformed_utterance = TfidfEncoder.encode_vectors(
            utterance, tfidf_vectorizer)
        transformed_utterance = NLUEngine.get_dense_array(classifier_model, transformed_utterance)

        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.inverse_transform(predicted_label)
        return decoded_label[0]

    @staticmethod
    def get_incorrect_predicted_labels():
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """
        #TODO: move this to the intent classifier class
        pass

    @staticmethod
    def encode_labels_and_utterances(
        data_df_path,
        labels_to_predict,
        classifier
        ):
        """
        Encode the labels and the utterances.
        :param data_df: pandas dataframe
        :return: intent classifier
        """
        #TODO: move this to the intent classifier class
        data_df = NLUEngine.load_data(data_df_path)

        if labels_to_predict == 'intent':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.intent.values)
        elif labels_to_predict == 'scenario' or labels_to_predict == 'domain':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.scenario.values)

        vectorized_utterances, tfidf_vectorizer = TfidfEncoder.encode_training_vectors(
            data_df)
        vectorized_utterances = NLUEngine.get_dense_array(
            classifier, vectorized_utterances)
        return encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer

    @staticmethod
    def train_intent_classifier(
        data_df_path,
        labels_to_predict,
        classifier
        ):
        """
        Train the intent classifier.
        :param data_df: pandas dataframe
        :return: intent classifier model
        """
        #NOTE: This method will stay in main and won't be moved to a separate class
        encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer = NLUEngine.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
        )
        return NLUEngine.train_classifier(classifier, vectorized_utterances, encoded_labels_to_predict), tfidf_vectorizer

    @staticmethod
    def evaluate_intent_classifier(
        data_df_path,
        labels_to_predict,
        classifier
    ):
        """
        Evaluates a classifier and generates a report
        """
        #NOTE: This method will stay in main and won't be moved to a separate class
        #TODO: rename this method to evaluate_intent_classifier
        print(f'Evaluating {classifier}')
        encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer = NLUEngine.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
            )

        
        predictions = Analytics.cross_validate_classifier(
            classifier,
            x_train=vectorized_utterances,
            y_train=encoded_labels_to_predict
        )

        predictions_decoded = LabelEncoder.decode(predictions).tolist()
        decoded_labels_to_predict = LabelEncoder.decode(
            encoded_labels_to_predict).tolist()
        report = Analytics.generate_report(
            classifier=classifier,
            predictions_decoded=predictions_decoded,
            decoded_labels_to_predict=decoded_labels_to_predict
        )

        report_df = Analytics.convert_report_to_df(
            classifier=classifier,
            report=report,
            label=labels_to_predict,
            encoding='tfidf'
        )
        return report_df

    @staticmethod
    def train_entity_classifier(data_df):
        """
        Train the entity classifier.
        :param data_df: pandas dataframe
        :return: entity classifier model
        """
        #TODO: check if data_df is a path or a dataframe, if path then load the dataframe
        print('Training entity classifier')
        X, y = EntityExtractor.get_targets_and_labels(data_df)
        crf_model = EntityExtractor.train_crf_model(X, y)
        return crf_model

    @staticmethod
    def create_entity_tagged_utterance(utterance, crf_model):
        """
        runs EntityExtractor.tag_utterance and gets all entity types and entities, formats the utterance into the entity annotated utterance
        This thing is a monster, it really should be refactored, perhaps it can be combined with other functions here to streamline the code
        """
        tagged_utterance = EntityExtractor.tag_utterance(utterance, crf_model)
        split_tagged_utterance = tagged_utterance.split(' ')

        for idx, token in enumerate(split_tagged_utterance):
            if '[' in token:
                if split_tagged_utterance[idx + 5]:
                    if token in split_tagged_utterance[idx + 3]:
                        split_tagged_utterance[idx + 2] = split_tagged_utterance[idx + 2].replace(
                            ']', '') + ' ' + split_tagged_utterance[idx + 5]

                        split_tagged_utterance[idx + 3] = ''
                        split_tagged_utterance[idx + 4] = ''
                        split_tagged_utterance[idx + 5] = ''

        """for idx, token in enumerate(split_tagged_utterance):
            if token is '':
                print (f'deleting {token} from {idx} position of {split_tagged_utterance}')
                del split_tagged_utterance[idx]"""

        normalised_split_tagged_utterance = [
            x for x in split_tagged_utterance if x]

        formatted_tagged_utterance = ' '.join(
            normalised_split_tagged_utterance)

        return formatted_tagged_utterance


    @staticmethod
    def evaluate_entity_classifier(data_df):
        """
        Evaluates the entity classifier and generates a report
        """

        print(f'Evaluating entity classifier')

        X, y = EntityExtractor.get_targets_and_labels(data_df)
        predictions = Analytics.cross_validate_classifier(crf, X, y)
        report_df = Analytics.generate_entity_classification_report(
            predictions, y)
        return report_df


'''nlu_engine_instance = NLUEngine()

DATA_PATH = 'NLU-Data-Home-Domain-Annotated-All.csv'
data_df = nlu_engine_instance.load_data(DATA_PATH)


domains = data_df.scenario.values

encoded_domains = nlu_engine_instance.encode_labels(domains)
#decoded_domains = nlu_engine_instance.decode_labels(encoded_domains)

vectorized_utterances = nlu_engine_instance.encode_tfidf_vectors(data_df)'''

"""
# This is to run the classifier to train a data set and predict new utterances
LR_model = nlu_engine_instance.train_classifier(
    nlu_engine_instance.LR,
    vectorized_utterances,
    encoded_domains
    )

utterance = "turn of the kitchen lights"

print(nlu_engine_instance.predict_label(LR_model, utterance))

"""

'''
This is to evalute a classifier and generate a report
'''

"""predictions = nlu_engine_instance.cross_validate_classifier(classifier=nlu_engine_instance.LR, x_train=vectorized_utterances, y_train=encoded_domains)

report = nlu_engine_instance.generate_report(classifier=nlu_engine_instance.LR, prediction=predictions, y_train=encoded_domains)

report_df = nlu_engine_instance.convert_report_to_df(classifier=nlu_engine_instance.LR, report=report, label='domain', encoding='tfidf')
print(report_df)"""

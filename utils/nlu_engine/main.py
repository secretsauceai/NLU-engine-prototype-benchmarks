import pandas as pd
import pickle


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
    def convert_annotated_utterances_to_utterances(data_df):
        """
        Convert the annotated utterances to the utterances.
        :param data_df: pandas dataframe
        :return: [string]
        """
        pass

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
    def predict_label(classifier_model, utterance):
        utterance = utterance.lower()
        print(f'Predicting label for utterance: {utterance}')
        transformed_utterance = TfidfEncoder.encode_vectors(utterance)
        print(f'Transformed utterance: {transformed_utterance}')
        transformed_utterance = NLUEngine.get_dense_array(classifier_model, transformed_utterance)
        print(f'Transformed utterance: {transformed_utterance}')
        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.inverse_transform(predicted_label)
        return decoded_label[0]

    @staticmethod
    def get_incorrect_predicted_labels():
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """
        pass

    @staticmethod
    def encode_labels_and_utterances(
        data_df_path,
        labels_to_predict,
        classifier
        ):
        """
        Train the intent classifier.
        :param data_df: pandas dataframe
        :return: intent classifier
        """
        data_df = NLUEngine.load_data(data_df_path)

        if labels_to_predict == 'intent':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.intent.values)
        elif labels_to_predict == 'scenario' or labels_to_predict == 'domain':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.scenario.values)

        vectorized_utterances = TfidfEncoder.encode_vectors(data_df)
        vectorized_utterances = NLUEngine.get_dense_array(
            classifier, vectorized_utterances)
        return encoded_labels_to_predict, vectorized_utterances

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
        encoded_labels_to_predict, vectorized_utterances = NLUEngine.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
        )
        return NLUEngine.train_classifier(classifier, vectorized_utterances, encoded_labels_to_predict)

    @staticmethod
    def evaluate_classifier(
        data_df_path,
        labels_to_predict,
        classifier
    ):
        """
        Evaluates a classifier and generates a report
        """
        print(f'Evaluating {classifier}')
        encoded_labels_to_predict, vectorized_utterances = NLUEngine.encode_labels_and_utterances(
            data_df_path,
            labels_to_predict,
            classifier
            )

        print(f'Encoded labels to predict: {encoded_labels_to_predict}')
        print(f'Vectorized utterances: {vectorized_utterances}')

        
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

    #TODO: create entity extraction class separately

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

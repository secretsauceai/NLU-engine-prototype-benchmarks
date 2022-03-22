from cProfile import label
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
 
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report

from .label_encoder import LabelEncoder
from .tfidf_encoder import TfidfEncoder
from .analytics import Analytics

#import nltk

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
    tfidf_vectorizer = None

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
    def predict_label(classifier_model, utterance):
        utterance = utterance.lower()
        transformed_utterance = NLUEngine.tfidf_vectorizer.transform([utterance])
        transformed_utterance = NLUEngine.get_dense_array(classifier_model, transformed_utterance)
        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.inverse_transform(predicted_label)
        return decoded_label[0]


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

        data_df = NLUEngine.load_data(data_df_path)

        if labels_to_predict == 'intent':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.intent.values)
        elif labels_to_predict == 'scenario' or labels_to_predict == 'domain':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.scenario.values)

        vectorized_utterances = TfidfEncoder.encode_vectors(data_df)
        vectorized_utterances = NLUEngine.get_dense_array(classifier, vectorized_utterances)
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

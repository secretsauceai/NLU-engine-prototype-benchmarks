from .label_encoder import LabelEncoder
from .tfidf_encoder import TfidfEncoder
from .data_utils import DataUtils
from .entity_extractor import EntityExtractor

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
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

class IntentMatcher:
    """
    All intent matching for the NLU engine is handled in the IntentMatcher.
    """
    @staticmethod
    def get_dense_array(classifier, x_train):
        """
        When using NB classifier, convert the utterances to a dense array.
        :param x_train: tfidf numpy array
        :return: tfidf dense numpy array
        """

        if classifier is NB:
            print(f'{NB} has been detected, switching to a dense array.')
            x_train = x_train.todense()
        else:
            pass
        return x_train

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

        data_df = DataUtils.load_data(data_df_path)

        encoded_labels_to_predict = LabelEncoder.encode(
                data_df[labels_to_predict].values) 

        vectorized_utterances, tfidf_vectorizer = TfidfEncoder.encode_training_vectors(
            data_df)
        vectorized_utterances = IntentMatcher.get_dense_array(
            classifier, vectorized_utterances)
        return encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer

    @staticmethod
    def train_classifier(classifier, x_train, y_train):
        # TODO: add in training time
        print(f'Training {str(classifier)}')
        x_train = IntentMatcher.get_dense_array(classifier, x_train)
        return classifier.fit(x_train, y_train)

    @staticmethod
    def predict_label(classifier_model, tfidf_vectorizer, utterance):
        """
        Predict the label of the utterance.
        :param classifier_model: classifier model
        :param tfidf_vectorizer: tfidf vectorizer
        :param utterance: string
        :return: label
        """
        if '[' in utterance:
            utterance = EntityExtractor.normalise_utterance(
                utterance=utterance)
        transformed_utterance = TfidfEncoder.encode_vectors(
            utterance, tfidf_vectorizer)
        transformed_utterance = IntentMatcher.get_dense_array(
            classifier_model, transformed_utterance)

        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.decode(predicted_label)
        return decoded_label[0]

    @staticmethod
    def get_incorrect_predicted_labels(data_df, classifier_model, tfidf_vectorizer):
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """

        output_df = data_df.copy()
        output_df['predicted_label'] = output_df['answer_normalised'].apply(
            lambda utterance:  IntentMatcher.predict_label(classifier_model, tfidf_vectorizer, utterance))
        return output_df[output_df['intent'] != output_df['predicted_label']]

    @staticmethod
    def get_predicted_labels(data_df, classifier_model, tfidf_vectorizer):
        """
        For a data set, get the predicted labels and return a dataframe.
        """

        output_df = data_df.copy()
        output_df['predicted_label'] = output_df['answer_normalised'].apply(
            lambda utterance:  IntentMatcher.predict_label(classifier_model, tfidf_vectorizer, utterance))
        return output_df

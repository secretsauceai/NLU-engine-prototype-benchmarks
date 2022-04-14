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
        data_df = DataUtils.load_data(data_df_path)

        if labels_to_predict == 'intent':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.intent.values)
        elif labels_to_predict == 'scenario' or labels_to_predict == 'domain':
            encoded_labels_to_predict = LabelEncoder.encode(
                data_df.scenario.values)

        vectorized_utterances, tfidf_vectorizer = TfidfEncoder.encode_training_vectors(
            data_df)
        vectorized_utterances = DataUtils.get_dense_array(
            classifier, vectorized_utterances)
        return encoded_labels_to_predict, vectorized_utterances, tfidf_vectorizer

    @staticmethod
    def train_classifier(classifier, x_train, y_train):
        # TODO: add in training time
        print(f'Training {str(classifier)}')
        x_train = DataUtils.get_dense_array(classifier, x_train)
        return classifier.fit(x_train, y_train)

    @staticmethod
    def predict_label(classifier_model, tfidf_vectorizer, utterance):
        #TODO: move this to the intent classifier class
        #utterance = utterance.lower()
        #print(f'Predicting label for utterance: {utterance}')
        # normalize the utterance without entity tags
        if '[' in utterance:
            utterance = EntityExtractor.normalise_utterance(
                utterance=utterance)
        transformed_utterance = TfidfEncoder.encode_vectors(
            utterance, tfidf_vectorizer)
        transformed_utterance = DataUtils.get_dense_array(
            classifier_model, transformed_utterance)

        predicted_label = classifier_model.predict(transformed_utterance)
        decoded_label = LabelEncoder.inverse_transform(predicted_label)
        return decoded_label[0]

    @staticmethod
    def get_incorrect_predicted_labels(data_df, classifier_model, tfidf_vectorizer):
        """
        For a data set, get the incorrect predicted labels and return a dataframe.
        """
        #TODO: implement in the analytics class
        output_df = data_df.copy()
        output_df['predicted_label'] = output_df['answer_normalised'].apply(
            lambda utterance:  IntentMatcher.predict_label(classifier_model, tfidf_vectorizer, utterance))
        return output_df[output_df['intent'] != output_df['predicted_label']]

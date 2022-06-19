import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

class Analytics:
    @staticmethod
    def cross_validate_classifier(classifier, x_train, y_train, cv=None):
        start = time.time()
        print(f'Cross validating with {str(classifier)}')
 
        prediction = cross_val_predict(
            estimator=classifier,
            X=x_train,
            y=y_train,
            cv=cv
        )
        stop = time.time()
        duration = stop - start
        print(f'Time it took to cross validate {str(classifier)}: {duration}')
        return prediction

    @staticmethod
    def generate_report(classifier, predictions_decoded, decoded_labels_to_predict):
        # TODO: change function to staticmethod

        report = classification_report(
            y_pred=predictions_decoded,
            y_true=decoded_labels_to_predict,
            output_dict=True
        )
        print(f'Generating report for {classifier}')
        return report

    @staticmethod
    def convert_report_to_df(classifier, report, label, encoding):
        """
        converts reports to a dataframe and adds the label of the prediction (i.e. intents or domains), encoding of the utterances (i.e. tfidf or word2vec), and the classifier used.
        """
        df = pd.DataFrame(report).transpose()
        df['classifier'] = str(classifier)
        df['classifier'] = df['classifier'].str.replace(r"\([^()]*\)", "")

        if label is 'intent':
            df.index = df.index.set_names(['intent'])
        elif label is 'scenario' or label is 'domain':
            df.index = df.index.set_names(['domain'])
        df = df.reset_index()
        output_df = df.copy()
        output_df['encoding'] = encoding
        return output_df

    @staticmethod
    def generate_entity_classification_report(predictions, y):
        """
        Generates a classification report for the entity classifier.
        """
        report = flat_classification_report(
            y_pred=predictions, y_true=y, output_dict=True)

        df = pd.DataFrame(report).transpose()
        df.index = df.index.set_names(['entity-type'])
        df = df.reset_index()
        return df

    @staticmethod
    def plot_report(report_df, improved_report_df=None):
        """
        Takes one or two in two report dataframes and plots the labels by their f1 scores.
        ::param report_df: A dataframe with the labels and f1 scores.
        ::param improved_report_df: A dataframe with the labels and f1 scores of a refined dataset. Can be None
        ::returns: A plot of the f1 scores
        """
        benchmark_df = report_df.drop(report_df.tail(3).index)
        label = report_df.columns[0]

        fig, ax = plt.subplots()

        if improved_report_df is not None:
            refined_benchmark_df = improved_report_df.drop(
                improved_report_df.tail(3).index)

            graph_report_df = pd.DataFrame.merge(
                benchmark_df, refined_benchmark_df, on=label, suffixes=('_original', '_refined'))

            graph_report_df.sort_values(
                by='f1-score_refined', ascending=True, inplace=True)

            graph_report_df.dropna(inplace=True)

            y_axis = np.arange(len(refined_benchmark_df[label]))
            bar_width = 0.35  # the width of the bars

            ax.barh(y_axis - bar_width/2,
                    graph_report_df['f1-score_original'], bar_width, label='Original', color='r')

            ax.barh(y_axis + bar_width/2,
                    graph_report_df['f1-score_refined'], bar_width, label='Refined', color='b')

        else:
            graph_report_df = benchmark_df.sort_values(
                by='f1-score', ascending=True)
            y_axis = np.arange(len(graph_report_df[label]))

            ax.barh(y_axis, graph_report_df['f1-score'],
                    align='center', color='b')

        fig.set_figheight(16)
        fig.set_figwidth(12)

        ax.set_title(f'f1-scores by {label}', fontsize=24)
        ax.set_xlabel("f1-score", fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_ylabel(label, fontsize=18)
        ax.set_yticks(y_axis, graph_report_df[label], fontsize=16)
        fig.tight_layout()

        if improved_report_df is not None:
            fig.savefig(f'data/reports/refined_{label}_report_graph.png')
        else:
            fig.savefig(f'data/reports/{label}_report_graph.png')

        plt.show()

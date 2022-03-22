from NLU_engine import NLUEngine

nlu_engine_instance = NLUEngine()

def evaluate_classifier(
    data_df_path,
    labels_to_predict,
    classifier
    ):
    """
    Evaluates a classifier and generates a report
    """
    print(f'Evaluating {classifier}')

    data_df = nlu_engine_instance.load_data(data_df_path)
    
    if labels_to_predict == 'intent':
        encoded_labels_to_predict = nlu_engine_instance.encode_labels(data_df.intent.values)
    elif labels_to_predict == 'scenario':
        encoded_labels_to_predict = nlu_engine_instance.encode_labels(data_df.scenario.values)

    vectorized_utterances = nlu_engine_instance.encode_tfidf_vectors(data_df)
    predictions = nlu_engine_instance.cross_validate_classifier(
        classifier,
        x_train=vectorized_utterances,
        y_train=encoded_labels_to_predict
        )

    report = nlu_engine_instance.generate_report(
    classifier=classifier,
    prediction=predictions,
        y_train=encoded_labels_to_predict
        )

    report_df = nlu_engine_instance.convert_report_to_df(
    classifier=nlu_engine_instance.LR,
    report=report,
    label=labels_to_predict,
    encoding='tfidf'
    )
    return report_df


'''DATA_PATH = 'NLU-Data-Home-Domain-Annotated-All.csv'


domain_labels = 'scenario'


report_df = evaluate_classifier(
    data_df_path=DATA_PATH,
    labels_to_predict=domain_labels,
    classifier=nlu_engine_instance.LR
    )

print(report_df)'''
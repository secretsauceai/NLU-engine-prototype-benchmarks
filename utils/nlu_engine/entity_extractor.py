import re
import nltk

import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report

class EntityExtractor:
    """
    Extracts entities from a text.
    """
    @staticmethod
    def seperate_types_and_entities(entities):
        """
        Seperates the entities into types and entities from the annotated utterance.
        """
        entity_list = []
        for entity in entities:
            split_entity = entity.split(' : ')
            entity_type = split_entity[0]
            entity_text = split_entity[1].split(' ')
            entity_list.append({'type': entity_type, 'words': entity_text})
        return entity_list

    @staticmethod
    def extract_entities(utterance):
        """
        Extracts entities from an annotated utterance (i.e. everything within square brackets:[]).
        """
        entities = re.findall(r'\[(.*?)\]', utterance)
        return EntityExtractor.seperate_types_and_entities(entities)

    @staticmethod
    def join_entities(utterance):
        """
        Join the extracted entities into a string.
        :param extracted_entities: list of entities
        :return: string
        """
        extracted_entities = EntityExtractor.extract_entities(utterance)

        entities_joined = str('')
        for idx, entity in enumerate(extracted_entities):
            entity_type = entity['type']
            if idx < len(extracted_entities) - 1:
                entities_joined = entities_joined + entity_type + '|'
            elif idx == len(extracted_entities) - 1:
                entities_joined = entities_joined + entity_type
        return entities_joined

    @staticmethod
    def normalise_utterance(utterance):
        """
            Normalise the utterance.
            :param utterance: string
            :return: string
            """
        entities_joined = EntityExtractor.join_entities(utterance)
        normalised_utterance = re.sub(entities_joined, '', utterance).replace(
            '[', '').replace(']', '').replace(' : ', '')
        return normalised_utterance

    @staticmethod
    def pos_tag_utterance(utterance):
        """
        POS tags a given utterance.
        """
        tokenized_utterance = nltk.word_tokenize(utterance)
        utterance_pos = nltk.pos_tag(tokenized_utterance)
        return utterance_pos


    @staticmethod
    def combine_pos_and_entity_tags(entities, utterance_pos):
        """
        Combines the POS tags with the entities.
        """
        output = []
        words = []

        for entity in entities:
            for word in entity['words']:
                words.append(word)

        for pair in utterance_pos:
            word = pair[0]
            pos = pair[1]
            for entity in entities:
                if word in entity['words']:
                    entity_type = entity['type']
                    output.append((word, pos, entity_type))
                elif word not in words and entity is entities[-1]:
                    entity_type = '0'
                    output.append((word, pos, entity_type))
        return output

    @staticmethod
    def create_feature_dataset(data_df):
        """
        Creates a feature dataset from the annotated utterances.
        """
        feature_dataset = []
        for utterance, utterance_with_tagging in zip(data_df['answer_normalised'], data_df['answer_annotation']):
            print(utterance)
            entities = EntityExtractor.extract_entities(utterance_with_tagging)
            utterance_pos = EntityExtractor.pos_tag_utterance(utterance)
            feature_dataset.append(
                EntityExtractor.combine_pos_and_entity_tags(entities, utterance_pos))
        return feature_dataset

    @staticmethod
    def create_crf_dataset(data_df):
        """
        Creates a dataset for the CRF model.
        """
        feature_dataset = EntityExtractor.create_feature_dataset(data_df)
        def word2features(utterance, i):
            word = utterance[i][0]
            postag = utterance[i][1]

            features = {
                'bias': 1.0,
                'word': word,
                'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],
                'postag': postag,
                'postag[:2]': postag[:2],
            }
            if i > 0:
                word1 = utterance[i-1][0]
                postag1 = utterance[i-1][1]
                features.update({
                    '-1:word': word1,
                    '-1:postag': postag1,
                    '-1:postag[:2]': postag1[:2],
                })
            else:
                features['BOS'] = True

            if i < len(utterance)-1:
                word1 = utterance[i+1][0]
                postag1 = utterance[i+1][1]
                features.update({
                    '+1:word': word1,
                    '+1:postag': postag1,
                    '+1:postag[:2]': postag1[:2],
                })
            else:
                features['EOS'] = True

            return features

        def utterance2features(utterance):
            return [word2features(utterance, i) for i in range(len(utterance))]

        def utterance2labels(utterance):
            return [label for token, postag, label in utterance]

        X = [utterance2features(utterance) for utterance in feature_dataset]
        y = [utterance2labels(utterance) for utterance in feature_dataset]

        return X, y

    @staticmethod
    def train_crf_model(X, y):
        """
        Trains a CRF model.
        """
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X, y)
        return crf
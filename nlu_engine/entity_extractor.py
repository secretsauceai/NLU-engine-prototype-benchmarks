import re
import nltk
import re

import sklearn_crfsuite

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

class EntityExtractor:
    """
    Extracts entities from an utterance.
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
    def tokenize_utterance(utterance):
        tokenized_utterance = nltk.word_tokenize(utterance)
        return tokenized_utterance

    @staticmethod
    def pos_tag_utterance(utterance):
        """
        POS tags a given utterance.
        """
        tokenized_utterance = EntityExtractor.tokenize_utterance(utterance)
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
            entities = EntityExtractor.extract_entities(utterance_with_tagging)
            utterance_pos = EntityExtractor.pos_tag_utterance(utterance)
            feature = EntityExtractor.combine_pos_and_entity_tags(
                entities, utterance_pos)
            feature_dataset.append(feature)
        return feature_dataset

    @staticmethod
    def word2features(utterance, i):
        """
        Creates the features for the CRF.
        """
        #TODO: Are the datasets correctly getting pos tags for all tokens (ie what's)?
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

    #TODO: add brown clustering features like https://github.com/leondz/entity_recognition/blob/f5ef5aafc259139b20c2d54dd54dc1f6f239b605/base_extractors.py#L41 or here https://github.com/aleju/ner-crf/blob/master/model/features.py

    @staticmethod
    def utterance2features(utterance):
        return [EntityExtractor.word2features(utterance, i) for i in range(len(utterance))]

    @staticmethod
    def utterance2labels(utterance):
        return [label for token, postag, label in utterance]

    @staticmethod
    def utterance2tokens(utterance):
        return [token for token, postag, label in utterance]

    @staticmethod
    def get_targets_and_labels(data_df):
        feature_dataset = EntityExtractor.create_feature_dataset(data_df)
        X = [EntityExtractor.utterance2features(utterance)
            for utterance in feature_dataset]
        y = [EntityExtractor.utterance2labels(utterance)
            for utterance in feature_dataset]

        return X, y


    @staticmethod
    def train_crf_model(X, y):
        """
        Trains a CRF model.
        """
        crf_model = crf.fit(X, y)
        return crf_model

    @staticmethod
    def predict_crf_model(crf_model, X):
        """
        Predicts the CRF model.
        """
        y_pred = crf_model.predict(X)
        return y_pred

    @staticmethod
    def get_entities(utterance, crf_model):
        utterance_pos = EntityExtractor.pos_tag_utterance(utterance)
        utterance_features = EntityExtractor.utterance2features(utterance_pos)
        label = crf_model.predict_single(utterance_features)
        return label

    @staticmethod
    def get_entity_types_and_locations(utterance, crf_model):
        entity_locations_and_types = []
        entities = EntityExtractor.get_entities(utterance, crf_model)
        for location, entity in enumerate(entities):
            if entity !="0":
                entity_locations_and_types.append((location, entity))
        return entity_locations_and_types

    @staticmethod
    def get_entity_tags(utterance, crf_model):
        entity_locations_and_types = EntityExtractor.get_entity_types_and_locations(utterance, crf_model)
        # TODO: Maybe not here?
        split_utterance = nltk.word_tokenize(utterance) #re.split(" |'", utterance)
        #tagged_entities = [(entity_type, split_utterance[location])
        #                for location, entity_type in entity_locations_and_types]
        
        tagged_entities = []

        for location, entity_type in entity_locations_and_types:
            '''
            # TODO: remove after finished with debugging
            if location not in split_utterance:
                print(f'{location}, failed for utterance: {split_utterance}')
            '''
            split_location = split_utterance[location]
            tagged_entities.append((entity_type, split_location))
        
        return tagged_entities

    @staticmethod
    def tag_utterance(utterance, crf_model):
        """
        replaces the entities with the tagged entities in the utterance like this example utterance:
        utterance = 'wake me up at five pm this week'
        tagged_utterance = 'wake me up at [time : five] [time : pm] [date : this] [date : week]'
        """
        #TODO: BUG! If you run: "who wrote the song i just wanna dance with you"
        # the output is: "who wrote the song [song_name : i] [song_name : just] wanna [song_name : dance] w[song_name : i]th [song_name : you]"
        tagged_entities = EntityExtractor.get_entity_tags(utterance, crf_model)
        tagged_utterance = utterance
        for entity_type, entity in tagged_entities:
            if entity + ' ' in utterance:
                tagged_utterance = tagged_utterance.replace(entity + ' ', "[{} : {}] ".format(entity_type, entity))
            elif ' ' + entity in utterance:
                tagged_utterance = tagged_utterance.replace(
                    ' ' + entity, " [{} : {}]".format(entity_type, entity))
                
        return tagged_utterance

    @staticmethod
    def get_incorrectly_tagged_entities(utterance, crf_model):
        """
        Returns a dataframe of incorrectly tagged entities.
        """
        #TODO: implement in the analytics class (or keep here?)
        pass
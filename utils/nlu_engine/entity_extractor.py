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
    def pos_tag_utterance(utterance):
        """
        POS tags a given utterance.
        """
        tokenized_utterance = nltk.word_tokenize(utterance)
        utterance_pos = nltk.pos_tag(tokenized_utterance)
        return utterance_pos

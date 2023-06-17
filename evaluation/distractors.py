import spacy
import random
from evaluation.conceptnet_utils import word_to_term, term_to_word, get_nodes

pronoun_groups = [
    ['i', 'we', 'you'],
    ['me', 'you', 'us', 'them', 'him', 'her'],
    ['he', 'she', 'they'],
    ['his', 'hers', 'theirs', 'mine', 'yours'],
    ['himself', 'herself', 'themselves', 'myself', 'yourself'],
]

nlp = spacy.load("en_core_web_lg")


def _find_pronoun_replacement(pronoun):
    for pronoun_group in pronoun_groups:
        if pronoun in pronoun_group:
            alternate_pronouns = list(set(pronoun_group) - set([pronoun]))
            new_pronoun = random.choice(alternate_pronouns)
            if new_pronoun == 'i':
                return 'I'
            return new_pronoun


def create_distractor(sentence):
    """
    returns a distractor sentence for the given sentence, by replacing a noun with a same-category noun
    """
    doc_dep = nlp(sentence)
    new_word = None
    for token in doc_dep:
        # for NOUN, try finding word in the same category
        noun_tokens = [t for t in doc_dep if t.pos_ in ['NOUN', 'PROPN']]
        random.shuffle(noun_tokens)
        while new_word is None and noun_tokens:
            token = noun_tokens[0]
            replaced_word = token.text
            category_terms = get_nodes(word_to_term(replaced_word), relation='IsA', node_type='end') or \
                            get_nodes(word_to_term(token.lemma_), relation='IsA', node_type='end')

            all_related_terms = {}
            for category_term, weight in category_terms.items():
                related_terms = get_nodes(category_term, relation='IsA', node_type='start')
                all_related_terms.update({k:weight for k,v in related_terms.items() 
                                          if (term_to_word(k) != replaced_word and '_' not in term_to_word(k))})
            
            if all_related_terms:
                new_term = max(all_related_terms, key=all_related_terms.get)
                if new_term:
                    new_word = term_to_word(new_term)
                    break
            noun_tokens = noun_tokens[1:]
    
    # if no replacement, replace pronoun with another (likely) grammatically correct pronoun
    if not new_word:
        for token in doc_dep:
            if token.pos_ == 'PRON':
                replaced_word = token.text
                new_word = _find_pronoun_replacement(replaced_word.lower())
                break
    
    # if still no replacement, replace random noun with 'corgi'
    if not new_word:
        noun_tokens = [t for t in doc_dep if t.pos_ in ['NOUN', 'PROPN', 'PRON']]
        replaced_word = random.choice(noun_tokens).text
        new_word = 'corgi'
    if not new_word:
        return sentence
    
    return sentence.replace(replaced_word, new_word, 1).capitalize()

import requests

def get_nodes(term, relation, node_type=None):
    """
    term: e.g., "/c/en/ask"
    relations: either a relation or a list of relations
    node_type: either 'start' or 'end'
    """
    if isinstance(relation, str):
        relation = [relation]
    if isinstance(node_type, str):
        node_type = [node_type]
    elif not node_type:
        node_type = ['start', 'end']
        
    obj = requests.get(f'http://api.conceptnet.io{term}?limit=1000').json()
    nodes = {}
    for e in obj['edges']:
        if e['rel']['label'] in relation:
            for n in node_type:
                node = e[n]
                if node['term'] != term:
                    nodes[node['term']] = e['weight']
    
    return nodes


def word_to_term(word):
    return f'/c/en/{word.lower()}'

def term_to_word(term):
    return term.split('/')[-1]
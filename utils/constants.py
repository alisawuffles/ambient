from pathlib import Path
import yaml

PRONOUNS = [
    "i", "you", "she", "he", "it", "we", "they", 
    "me", "her", "him", "us", "them", 
    "hers", "his", 'my', 'mine', 'your', 'yours', 'their', 'theirs', 
    'this', 'that', 'the', 'anyone', 'anybody', 'everyone',
    "i'm", "i am", "she's", "she is", "he's", "he is", "they're", "they are", "we are", "we're"
]

# Config
CONFIG_FILE = Path('config.yml')
OPENAI_API_KEY = ''
try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')

# NLI things
NLI_LABELS = ['contradiction', 'entailment', 'neutral']
id2label = {i: label for i, label in enumerate(NLI_LABELS)}
id2label[3] = 'discard'
label2id = {l:i for i,l in id2label.items()}
SEED = 1
MAX_DISAMBIGUATIONS = 4
NLI_SENTENCE_KEYS = ['premise', 'hypothesis']
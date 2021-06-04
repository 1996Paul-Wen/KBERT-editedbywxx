import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'Medical': os.path.join(FILE_DIR_PATH, 'kgs/Medical.spo'),
    'CnDbandMedical': os.path.join(FILE_DIR_PATH, 'kgs/CnDbandMedical.spo'),
    'Medical-plus': os.path.join(FILE_DIR_PATH, 'kgs/Medical-plus.spo'),
    'Medical-plus_nocheck': os.path.join(FILE_DIR_PATH, 'kgs/Medical-plus_nocheck.spo'),
}

MAX_ENTITIES = 1

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

# 以上token均不可被split
NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]

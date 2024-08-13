TRAIN_FILE_PATH = 'data/train.txt'
VAL_FILE_PATH = 'data/val.txt'
TEST_FILE_PATH = 'data/test.txt'

CHARACTERS_FILE_PATH = 'data/characters.pkl'

EMPTY = ''
PAD = '<pad>'
END = '<end>'

FATHA = 'َ'
DAMMA = 'ُ'
KASRA = 'ِ'
FATHATAN = 'ً'
DAMMATAN = 'ٌ'
KASRATAN = 'ٍ'
SHADDA = 'ّ'
SUKUN = 'ْ'

DIACRITICS = [
    FATHA,
    DAMMA,
    KASRA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    SHADDA,
]

DIACRITICS_COMBINATIONS = [
    FATHA,
    DAMMA,
    KASRA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    SHADDA,
    SHADDA + FATHA,
    SHADDA + DAMMA,
    SHADDA + KASRA,
    SHADDA + FATHATAN,
    SHADDA + DAMMATAN,
    SHADDA + KASRATAN,
    SUKUN,
    EMPTY,
    PAD,
    END,
]

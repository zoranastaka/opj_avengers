import pandas as pd
import re
import sys
from Croatian_stemmer import Croatian_stemmer as stemmer

RULES=[
    re.compile(r'^('+osnova+')('+nastavak+r')$')
    for osnova, nastavak in [e.strip().split(' ') for e in open('Croatian_stemmer/rules.txt')]
]

TRANSFORMATIONS = [e.strip().split('\t') for e in open('Croatian_stemmer/transformations.txt')]

STOP = {
    'biti','jesam','budem','sam','jesi','budeš','si',
    'jesmo','budemo','smo','jeste','budete','ste','jesu',
    'budu','su','bih','bijah','bjeh','bijaše','bi','bje',
    'bješe','bijasmo','bismo','bjesmo','bijaste','biste',
    'bjeste','bijahu','biste','bjeste','bijahu','bi',
    'biše','bjehu','bješe','bio','bili','budimo','budite',
    'bila','bilo','bile','ću','ćeš','će','ćemo','ćete',
    'želim','želiš','želi','želimo','želite','žele','moram',
    'moraš','mora','moramo','morate','moraju','trebam',
    'trebaš','treba','trebamo','trebate','trebaju','mogu',
    'možeš','može','možemo','možete'
}


def transform_annotated_dataframe(annotated_df):
    """
    Transform annotated dataframe into format required by specification
    """

    lista_uzoraka = []
    upiti = [column.strip() for column in list(annotated_df.columns)[4:]]

    for i, row in annotated_df.iterrows():
        for upitid, upit in enumerate(upiti[1:]):
            lista_uzoraka.append([row['pair_id'], upitid, row['Komentar'], upit, row[upit]])

    df = pd.DataFrame(
        lista_uzoraka, columns=['PairID', 'QueryID', 'Comment', 'Query', 'Score']
    ).astype(
        {
            'PairID': str,
            'QueryID': int,
            'Comment': str,
            'Query': str,
            'Score': int
        }
    )

    return df


def normalize_to_lowercase(df, inplace=False, columns={'Comment', 'Query'}):
    """
    Normalize desired columns to lowercase.

    :param:
    """

    if not inplace:
        temp_df = df.copy()
        for column in columns:
            temp_df[column] = temp_df[column].apply(lambda x: x.lower())
        return temp_df
    else:
        for column in columns:
            df[column] = df[column].apply(lambda x: x.lower())
    return


def tokenize_croatian(text):
    """
    Tokenizes text using Croatian stemmer - http://nlp.ffzg.hr/resources/tools/stemmer-for-croatian/
    :param text: Input text which will be tokenized.
    """
    new_text = ''

    for token in re.findall(r'\w+', text, re.UNICODE):
        if token.lower() in STOP:
            try:
                new_text += token.lower() + ' '
            except TypeError:
                print(type(token))
                print(token)
                raise TypeError
            continue
        new_text += stemmer.korjenuj(
            stemmer.transformiraj(token.lower(), TRANSFORMATIONS), RULES
        ) + ' '
    return new_text


def apply_tokenize(df, tokenize=tokenize_croatian, inplace=True, replace_columns=True, columns={'Comment', 'Query'}):

    if replace_columns:
        column_dict = {column:column for column in columns}
    else:
        column_dict = {column:column + 'Stem' for column in columns}

    if not inplace:
        temp_df = df.copy()
        for column in columns:
            temp_df[column_dict[column]] = temp_df[column].apply(lambda x: tokenize(x))
        return temp_df
    else:
        for column in columns:
            df[column_dict[column]] = df[column].apply(lambda x: tokenize(x))
    return


def count_common_words(text1, text2, remove_duplicates=True):

    words_text1 = set(text1.strip().split(' '))
    if remove_duplicates:
        words_text2 = set(text2.strip().split(' '))
    else:
        words_text2 = text2.strip().split(' ')

    word_counter = 0
    for word in words_text2:
        if word in words_text1:
            word_counter += 1
    return word_counter





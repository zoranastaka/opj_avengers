import pandas as pd
import re
import sys

from opj_avengers.Croatian_stemmer import Croatian_stemmer as stemmer

# Transformisanje podataka u oblik prigodan za klasifikaciju

df_merged = pd.read_csv('../../merged_all.csv')

descriptive_columns = ['pair_id', 'comment', 'Komentar', 'code']

queries = df_merged.drop(columns=descriptive_columns, inplace=False)
lista_uzoraka = []
upiti = [column.strip() for column in queries.columns]
df_merged.rename(columns={column: column.strip() for column in queries.columns}, inplace=True)

for i, row in df_merged.iterrows():
    for upitid, upit in enumerate(upiti[1:]):
        lista_uzoraka.append([row['pair_id'], upitid, row['Komentar'], upit, row[upit]])

df_uzorci = pd.DataFrame(lista_uzoraka, columns=['PairID', 'QueryID', 'Komentar', 'Query', 'Score']).astype({'PairID': str, 'QueryID': int, 'Komentar': str, 'Query': str, 'Score': int})

df_uzorci.head(3)

df_uzorci[['PairID', 'QueryID', 'Komentar', 'Query', 'Score']].to_csv('Pair-query-score.csv', index=False)

# Predobrada podataka

# Normalizacija tekstova na mala slova

df_uzorci['Komentar'] = df_uzorci['Komentar'].apply(lambda x: x.lower())
df_uzorci['Query'] = df_uzorci['Query'].apply(lambda x: x.lower())

# Stemovanje reči
# Stemovanje se vrši stemerom za hrvatski jezik

pravila=[
    re.compile(r'^('+osnova+')('+nastavak+r')$')
    for osnova, nastavak in [e.strip().split(' ') for e in open('Croatian_stemmer/rules.txt')]
]

transformacije = [e.strip().split('\t') for e in open('Croatian_stemmer/transformations.txt')]

stop = set(['biti','jesam','budem','sam','jesi','budeš','si','jesmo','budemo','smo','jeste','budete','ste','jesu','budu','su','bih','bijah','bjeh','bijaše','bi','bje','bješe','bijasmo','bismo','bjesmo','bijaste','biste','bjeste','bijahu','biste','bjeste','bijahu','bi','biše','bjehu','bješe','bio','bili','budimo','budite','bila','bilo','bile','ću','ćeš','će','ćemo','ćete','želim','želiš','želi','želimo','želite','žele','moram','moraš','mora','moramo','morate','moraju','trebam','trebaš','treba','trebamo','trebate','trebaju','mogu','možeš','može','možemo','možete'])

def tokenize(text):
    new_text = ''

    for token in re.findall(r'\w+',text ,re.UNICODE):
        if token.lower() in stop:
            try:
                new_text += token.lower() + ' '
            except TypeError:
                print(type(token))
                print(token)
                raise(TypeError)
            continue
        new_text += stemmer.korjenuj(
            stemmer.transformiraj(token.lower(), transformacije), pravila
        ) + ' '
    return new_text

df_uzorci['Stem'] = df_uzorci['Komentar'].apply(lambda x: tokenize(x))
df_uzorci['StemUpit'] = df_uzorci['Query'].apply(lambda x: tokenize(x))
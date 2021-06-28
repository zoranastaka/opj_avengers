# %%

import numpy as np
import pandas as pd

# %%

df_Boris = pd.read_csv('data/Boris/Boris_anotirano_v2.csv')
boris_columns = df_Boris.columns

df_Branislav = pd.read_csv('data/Branislav/annotation_matrix_Branislav.csv', names=boris_columns, header=0)
df_Zorana = pd.read_csv('data/Zorana/annotation_matrix_Zorana_konacno.csv', names=boris_columns, header=0)
df_Nikola = pd.read_csv('data/Nikola/anotaciona_konacna2_Nikola.csv', names=boris_columns, header=0)

# %%

column_rename = {column_nikola: column_original for column_original, column_nikola in zip(df_Boris.columns, df_Nikola.columns)}
df_Nikola.rename(columns=column_rename, inplace=True)

# %%

df_merged = df_Boris.append(df_Branislav).append(df_Zorana).append(df_Nikola)

df_merged = df_merged.dropna()


descriptive_columns = ['pair_id', 'comment', 'Komentar', 'code']
queries = df_merged #.drop(columns=descriptive_columns, inplace=False)
queries.head(1)

# %%

query_count = queries.fillna(0).astype(bool).sum(axis=0)

# %%

rare_queries = query_count[query_count < 1]
len(rare_queries)


import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3), dpi=160)
plt.hist(query_count.values, bins=25)
plt.xlabel('Broj pojavljivanja ne-nultih skorova')
plt.ylabel('Broj upita')
plt.title('Histogram broja upita anotiranih nenultim skorom sličnosti')
plt.grid(alpha=0.2)

plt.show()

# %%

len(df_merged.columns)

# %%

df_merged.isna()

# %% md

# Priprema za klasifikaciju

# %%

# Transformisanje podataka u oblik prigodan za klasifikaciju

lista_uzoraka = []
upiti = [column.strip() for column in queries.columns]
df_merged.rename(columns={column: column.strip() for column in queries.columns}, inplace=True)
for i, row in df_merged.iterrows():
    for upitid, upit in enumerate(upiti[1:]):
        lista_uzoraka.append([row['pair_id'], upitid, row['Komentar'], upit, row[upit]])

df_uzorci = pd.DataFrame(lista_uzoraka, columns=['PairID', 'QueryID', 'Komentar', 'Upit', 'Vrednost']).astype({'PairID': str, 'QueryID': int, 'Komentar': str, 'Upit': str, 'Vrednost': int})

df_uzorci.head(3)

# %% md

## Predobrada podataka

### Normalizacija tekstova na mala slova

# %%

df_uzorci['Komentar'] = df_uzorci['Komentar'].apply(lambda x: x.lower())
df_uzorci['Upit'] = df_uzorci['Upit'].apply(lambda x: x.lower())

# %% md

### Stemovanje reči


# %%

import re
from Croatian_stemmer import Croatian_stemmer as stemmer

# %%

pravila = [re.compile(r'^(' + osnova + ')(' + nastavak + r')$') for osnova, nastavak in [e.strip().split(' ') for e in open('Croatian_stemmer/rules.txt')]]

transformacije = [e.strip().split('\t') for e in open('Croatian_stemmer/transformations.txt')]

stop = set(['biti', 'jesam', 'budem', 'sam', 'jesi', 'budeš', 'si', 'jesmo', 'budemo', 'smo', 'jeste', 'budete', 'ste', 'jesu', 'budu', 'su', 'bih', 'bijah', 'bjeh', 'bijaše', 'bi', 'bje', 'bješe', 'bijasmo', 'bismo', 'bjesmo', 'bijaste', 'biste', 'bjeste', 'bijahu', 'biste', 'bjeste', 'bijahu', 'bi', 'biše', 'bjehu', 'bješe', 'bio', 'bili', 'budimo', 'budite', 'bila', 'bilo', 'bile', 'ću', 'ćeš', 'će', 'ćemo', 'ćete', 'želim', 'želiš', 'želi', 'želimo', 'želite', 'žele', 'moram', 'moraš', 'mora', 'moramo', 'morate', 'moraju', 'trebam', 'trebaš', 'treba', 'trebamo', 'trebate', 'trebaju', 'mogu', 'možeš', 'može', 'možemo', 'možete'])


def tokenize(text):
    new_text = ''

    for token in re.findall(r'\w+', text, re.UNICODE):
        if token.lower() in stop:
            try:
                new_text += token.lower() + ' '
            except TypeError:
                print(type(token))
                print(token)
                raise (TypeError)
            continue
        new_text += stemmer.korjenuj(stemmer.transformiraj(token.lower(), transformacije), pravila) + ' '
    return new_text


# %%

df_uzorci['Stem'] = df_uzorci['Komentar'].apply(lambda x: tokenize(x))
df_uzorci['StemUpit'] = df_uzorci['Upit'].apply(lambda x: tokenize(x))

# %%

df_uzorci.head(3)

# %% md

### Filtriranje po frekvenciji reci

# %%

word_dict = {}
for _, row in df_merged.iterrows():
    try:
        for word in tokenize(row['Komentar'].strip()).split(' '):
            if word.isnumeric() or len(word) < 2:
                continue
            if not word in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
            if word == 'inicijaliz':
                print(row)
    except:
        continue

for query in df_merged.columns:
    try:
        for word in tokenize(query).strip().split(' '):
            if word.isnumeric() or len(word) < 2:
                continue
            if not word in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
            if word == 'inicijaliz':
                print(row)
    except:
        continue

query_word_dict = {}
for query in df_merged.columns:
    for word in tokenize(query).split(' '):
        if word.isnumeric() or len(word) < 2:
            continue
        if not word in query_word_dict:
            query_word_dict[word] = 1
        else:
            query_word_dict[word] += 1

# %%


# %%

# Koliko imamo jedinstvenih reči
len(word_dict)

# %%

word_lengths = list(word_dict.values())
word_lengths.sort()

# %%

# Izdvajanje retkih i čestih reči
rare_words = [word for word in word_dict.keys() if word_dict[word] <= word_lengths[round(0.1 * len(word_dict))]]
common_words = [word for word in word_dict.keys() if word_dict[word] >= word_lengths[round(0.95 * len(word_dict))]]

# One česte reči koje su delovi upita zadržavamo i koje imaju više od dva slova. Veznike ipak želimo da izbacimo
common_words = [word for word in common_words if word not in query_word_dict or len(word) < 3]

# %%

common_words[:10]


# %%

def broj_istih_reci_bez_ponavljanja(tekst1, tekst2):
    reci_tekst1 = set(tekst1.strip().split(' '))
    reci_tekst2 = set(tekst2.strip().split(' '))
    word_counter = 0
    for rec in reci_tekst1:
        if rec in reci_tekst2:
            word_counter += 1
    #             print("{} - {} - {}".format(tekst1, tekst2, rec))
    return word_counter


# %%

def broj_istih_reci_sa_ponavljanjem(tekst1, tekst2):
    reci_tekst1 = set(tekst1.strip().split(' '))
    reci_tekst2 = tekst2.strip().split(' ')
    word_counter = 0
    for rec in reci_tekst2:
        if rec in reci_tekst1:
            word_counter += 1
    return word_counter


# %%

def broj_zajednickih_bigrama(tekst1, tekst2):
    reci_t1 = tekst1.strip().split(' ')
    bigrami_t1 = [reci_t1[i] + ' ' + reci_t1[i + 1] for i in range(len(reci_t1) - 1)]
    reci_t2 = tekst2.strip().split(' ')
    bigrami_t2 = [reci_t2[i] + ' ' + reci_t2[i + 1] for i in range(len(reci_t2) - 1)]
    counter = 0
    for bigram in bigrami_t1:
        if bigram in bigrami_t2:
            counter += 1

    return counter


# %%

def ukloni_ceste_reci(tekst, ceste_reci):
    reci = [rec for rec in tekst.strip().split(' ') if rec not in ceste_reci]
    return ' '.join(reci)


# %%

df_uzorci['StemFiltered'] = df_uzorci['Stem'].apply(lambda x: ukloni_ceste_reci(x, common_words))
df_uzorci['StemUpitFiltered'] = df_uzorci['StemUpit'].apply(lambda x: ukloni_ceste_reci(x, common_words))
df_uzorci['ZajednickeStem'] = df_uzorci.apply(lambda x: broj_istih_reci_bez_ponavljanja(x['Stem'], x['StemUpit']), axis=1)
df_uzorci['ZajednickeOriginal'] = df_uzorci.apply(lambda x: broj_istih_reci_bez_ponavljanja(x['Komentar'], x['Upit']), axis=1)

df_uzorci['Bigrami'] = df_uzorci.apply(lambda x: broj_zajednickih_bigrama(x['Stem'], x['StemUpit']), axis=1)
df_uzorci['BrojReciStem'] = df_uzorci['Stem'].apply(lambda x: len(x.strip().split(' ')))
df_uzorci['BrojReciUpit'] = df_uzorci['StemUpit'].apply(lambda x: len(x.strip().split(' ')))

# %%

df_uzorci.head(3)

# %% md

## TF - IDF - TIFDF

# %%

unique_words = list(word_dict.keys())

# %%

for key in word_dict:
    if word_dict[key] == 0:
        print(word_dict)

# %%

bag_of_words = {}
for i, row in df_merged.iterrows():
    current_words = tokenize(row['Komentar']).split()
    bag_of_words[row['pair_id']] = [sum([a == word for a in current_words]) for word in unique_words]

# %%

df_BOW = pd.DataFrame.from_dict(bag_of_words, orient='index', columns=unique_words)
df_BOW.head(3)

# %%

upiti = [query.strip() for query in queries.columns][1:]
bag_of_words_upiti = {}
for i, upit in enumerate(upiti):
    current_words = tokenize(upit).split()
    bag_of_words_upiti[i] = [sum([a == word for a in current_words]) for word in unique_words]

# %%

df_BOWQ = pd.DataFrame.from_dict(bag_of_words_upiti, orient='index', columns=unique_words)
df_BOWQ.head(3)

# %%

word_occurances = [(sum(df_BOW[word] > 0) + sum(df_BOWQ[word] > 0)) for word in unique_words]
words_to_drop = [unique_words[i] for i, broj in enumerate(word_occurances) if broj == 0]

# %%

df_BOWQ.drop(words_to_drop, inplace=True, axis=1)
df_BOW.drop(words_to_drop, inplace=True, axis=1)
unique_words = [word for word in unique_words if word not in words_to_drop]

# %%

df_BOW.head()

# %%

df_BOWQ.head()

# %%

tf = {}
for key in bag_of_words:
    curr_sum = sum(df_BOW.loc[key])
    tf[key] = [c / curr_sum for c in df_BOW.loc[key].values]

# %%

tf_df = pd.DataFrame.from_dict(tf, orient='index', columns=unique_words)

# %%

tf_df.head()

# %%

tf_upiti = {}

for key in bag_of_words_upiti:
    curr_sum = sum(df_BOWQ.loc[key])
    tf_upiti[key] = [c / curr_sum for c in df_BOWQ.loc[key].values]

# %%

tf_dfQ = pd.DataFrame.from_dict(tf_upiti, orient='index', columns=unique_words)
tf_dfQ.head(3)

# %%

number_of_documents = len(df_merged)
idf = [np.log(number_of_documents / (sum(df_BOW[word] > 0) + sum(df_BOWQ[word] > 0))) for word in unique_words]

# %%

idf[0:10]

# %%

tfidf = tf_df.copy().multiply(idf, axis=1)
tfidf.head()

# %%

tfidfQ = tf_dfQ.copy().multiply(idf, axis=1)
tfidfQ.head()


# %%

# Sada treba napraviti obeležja od kosinusnih proizvoda

# %%

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# %%

df_uzorci['BOW'] = df_uzorci.apply(lambda x: cosine_sim(df_BOW.loc[x['PairID']], df_BOWQ.loc[x['QueryID']]), axis=1)

# %%

df_uzorci['TF'] = df_uzorci.apply(lambda x: cosine_sim(tf_df.loc[x['PairID']], tf_dfQ.loc[x['QueryID']]), axis=1)

# %%

df_uzorci['TFIDF'] = df_uzorci.apply(lambda x: cosine_sim(tf_df.loc[x['PairID']], tf_dfQ.loc[x['QueryID']]), axis=1)

# %%

df_uzorci.to_csv('All_Data.csv')

# %%

len(df_uzorci['Upit'].unique())

# %%

df_uzorci.head()

# %%

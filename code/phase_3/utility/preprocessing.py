import pandas as pd
import numpy as np
import re
import sys
from Croatian_stemmer import Croatian_stemmer as stemmer

RULES = [re.compile(r'^(' + osnova + ')(' + nastavak + r')$') for osnova, nastavak in [e.strip().split(' ') for e in open('Croatian_stemmer/rules.txt', encoding="utf8")]]

TRANSFORMATIONS = [e.strip().split('\t') for e in open('Croatian_stemmer/transformations.txt', encoding="utf8")]

STOP = {'biti', 'jesam', 'budem', 'sam', 'jesi', 'budeš', 'si', 'jesmo', 'budemo', 'smo', 'jeste', 'budete', 'ste', 'jesu', 'budu', 'su', 'bih', 'bijah', 'bjeh', 'bijaše', 'bi', 'bje', 'bješe', 'bijasmo', 'bismo', 'bjesmo', 'bijaste', 'biste', 'bjeste', 'bijahu', 'biste', 'bjeste', 'bijahu', 'bi', 'biše', 'bjehu', 'bješe', 'bio', 'bili', 'budimo', 'budite', 'bila', 'bilo', 'bile', 'ću', 'ćeš', 'će', 'ćemo', 'ćete', 'želim', 'želiš', 'želi', 'želimo', 'želite', 'žele', 'moram', 'moraš', 'mora', 'moramo', 'morate', 'moraju', 'trebam', 'trebaš', 'treba', 'trebamo', 'trebate', 'trebaju', 'mogu', 'možeš', 'može', 'možemo', 'možete', "a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "biste", "bumo", "da", "do", "duž", "ga", "hoće", "hoćemo", "hoćete", "hoćeš", "hoću", "i", "iako", "ih", "ili", "iz", "ja", "je", "jedna", "jedne", "jedno", "jer", "jest", "jesu", "joj",
    "još", "ju", "kada", "kako", "kao", "koja", "koje", "koji", "kojima", "koju", "kroz", "li", "me", "mene", "meni", "mi", "mimo", "moj", "moja", "moje", "mu", "na", "nad", "nakon", "nam", "nama", "nas", "naš", "naša", "naše", "našeg", "ne", "nego", "neka", "neki", "nekog", "neku", "nema", "netko", "neće", "nećemo", "nećete", "nećeš", "neću", "nešto", "ni", "nije", "nikoga", "nikoje", "nikoju", "nisam", "nisi", "nismo", "niste", "nisu", "njega", "njegov", "njegova", "njegovo", "njemu", "njezin", "njezina", "njezino", "njih", "njihov", "njihova", "njihovo", "njim", "njima", "njoj", "nju", "no", "o", "od", "odmah", "on", "ona", "oni", "ono	", "ova", "pa", "pak", "po", "pod", "pored", "prije", "s", "sa", "samo", "se", "sebe", "sebi", "si", "smo", "ste", "su", "sve", "svi", "svog", "svoj", "svoja", "svoje", "svom", "ta", "tada", "taj", "tako", "te", "tebe", "tebi", "ti",
    "to", "tome", "tu", "tvoj", "tvoja", "tvoje", "u", "uz", "vam", "vama", "vas", "vaš", "vaša", "vaše", "već", "vi", "vrlo", "za", "zar", "će", "ćemo", "ćete", "ćeš", "ću", "što"}


def transform_annotated_dataframe(annotated_df):
    """
    Transform annotated dataframe into format required by specification
    """

    lista_uzoraka = []
    upiti = [column.strip() for column in list(annotated_df.columns)[4:]]

    for i, row in annotated_df.iterrows():
        for upitid, upit in enumerate(upiti[1:]):
            lista_uzoraka.append([row['pair_id'], upitid, row['Komentar'], upit, row[upit]])

    df = pd.DataFrame(lista_uzoraka, columns=['PairID', 'QueryID', 'Comment', 'Query', 'Score']).astype({'PairID': str, 'QueryID': int, 'Comment': str, 'Query': str, 'Score': int})

    return df


def normalize_to_lowercase(df, inplace=False, columns={'Comment', 'Query'}):
    """
    Normalize desired columns to lowercase.

    :param df: Dataframe containing unique comment-query rows.
    :param inplace: If True modify df. If False, return a new dataframe. default: False
    :param columns: Array-like collection of columns to which normalization will be applied. default: {'Comment', 'Query'}
    :return: If inplace is False, returns a copy of dataframe with normalized columns. Otherwise, performs normalization in place.
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


def stem_croatian(text, ignore_case=False):
    """
    Stems text using Croatian stemmer - http://nlp.ffzg.hr/resources/tools/stemmer-for-croatian/

    :param text: Input text which will be tokenized.
    :param ignore_case: Don't make difference between lower and upper cases
    :return: Text, but all words stemmed using Croatian stemmer.
    """
    new_text = ''
    for token in re.findall(r'\w+', text, re.UNICODE):
        if token.lower() in STOP:
            new_text += (token if ignore_case else token.lower()) + ' '
            continue

        temp = stemmer.korjenuj(stemmer.transformiraj(token.lower(), TRANSFORMATIONS), RULES)
        if ignore_case:
            temp = token[0:temp.__len__()]

        new_text += temp + ' '

    return new_text


def apply_stem(df, tokenize=stem_croatian, inplace=True, replace_columns=True, columns={'Comment', 'Query'}):
    """
        Stems desired columns using a specified stemmer.

        :param df: Dataframe containing unique comment-query rows.
        :param tokenize: Tokenization functions to use.
        :param inplace: If True modify df. If False, return a new dataframe. default: False
        :param replace_columns: If True, specified columns will be modified. If False, new columns with suffix 'Stem' will be generated. default: True
        :param columns: Array-like collection of columns to which stemming will be applied. default: {'Comment', 'Query'}
        :return: If inplace is False, returns a copy of dataframe with stemmed columns. Otherwise, performs normalization in place.
        """

    if replace_columns:
        column_dict = {column: column for column in columns}
    else:
        column_dict = {column: column + 'Stem' for column in columns}

    if not inplace:
        temp_df = df.copy()
        for column in columns:
            temp_df[column_dict[column]] = temp_df[column].apply(lambda x: tokenize(x))
        return temp_df
    else:
        for column in columns:
            df[column_dict[column]] = df[column].apply(lambda x: tokenize(x))
    return


def count_mutual_words(text1, text2, remove_duplicates=True):
    """
    Counts number of words which occur in both texts.

    :param text1: First text.
    :param text2: Second text.
    :param remove_duplicates: If True, multiple occurancies of words will still be counted as one word. Note that if
    one word is present multiple times in both texts, only multiple occurancies in text2 are counted.
    :return: Number of words occurring in both texts.
    """

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


def count_mutual_bigrams(text1, text2):
    """
    Counts number of bigrams which occur in both texts.

    :param text1: First text
    :param text2: Second text
    :return:
    """

    words_text1 = text1.strip().split(' ')
    bigrams_text1 = [words_text1[i] + ' ' + words_text1[i + 1] for i in range(len(words_text1) - 1)]
    words_text2 = text2.strip().split(' ')
    bigrams_text2 = [words_text2[i] + ' ' + words_text2[i + 1] for i in range(len(words_text2) - 1)]
    counter = 0
    for bigram in bigrams_text1:
        if bigram in bigrams_text2:
            counter += 1
    return counter


def count_mutual_trigrams(text1, text2):
    """
    Counts number of trigrams which occur in both texts.

    :param text1: First text
    :param text2: Second text
    :return:
    """

    words_text1 = text1.strip().split(' ')
    trigrams_text1 = [words_text1[i] + ' ' + words_text1[i + 1] + ' ' + words_text1[i + 2] for i in range(len(words_text1) - 2)]
    words_text2 = text2.strip().split(' ')
    trigrams_text2 = [words_text2[i] + ' ' + words_text2[i + 1] + ' ' + words_text2[i + 2] for i in range(len(words_text2) - 2)]

    counter = 0
    for trigram in trigrams_text1:
        if trigram in trigrams_text2:
            counter += 1
    return counter


def make_word_dict(df, minimal_length=2, comment_column='Comment', query_column='Query'):
    """

    :param df:
    :param minimal_length:
    :return:
    """
    word_dict = {}
    for comment in df[comment_column].unique():
        try:
            for word in comment.strip().split():
                if word.isnumeric() or len(word) < minimal_length:
                    continue
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        except:
            continue

    queries = df[query_column].unique()
    for query in queries:
        try:
            for word in query.strip().split():
                if word.isnumeric() or len(word) < minimal_length:
                    continue
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        except:
            continue

    query_word_dict = {}
    for query in queries:
        for word in query.strip().split():
            if word.isnumeric() or len(word) < minimal_length:
                continue
            if word not in query_word_dict:
                query_word_dict[word] = 1
            else:
                query_word_dict[word] += 1

    return {'All': word_dict.copy(), 'Query': query_word_dict.copy()}


def remove_common_words(text, common_words):
    """
    Removes common words from text.

    :param text: Text from which words will be removed.
    :param common_words: Array-like common words.
    :return: Text without common words (string).
    """
    words = [word for word in text.strip().split(' ') if word not in common_words]
    return ' '.join(words)


def remove_rare_words(text, rare_words):
    """
    Removes common words from text.

    :param text: Text from which words will be removed.
    :param rare_words: Array-like rare words.
    :return: Text without rare words (string)
    """

    words = [word for word in text.strip().split(' ') if word not in rare_words]
    return ' '.join(words)


def remove_interpunction(text):
    """
    Removes interpunction from text
    :param text: Text from which interpunction will be removed.
    :return: Text without interpunction
    """

    new_text = text.replace(', ', ' ')
    new_text = new_text.replace('. ', ' ')
    new_text = new_text.replace('; ', ' ')
    new_text = new_text.replace('-', ' ')
    new_text = new_text.replace('! ', ' ')
    new_text = new_text.replace('? ', ' ')
    new_text = new_text.replace('"', '')
    new_text = new_text.replace('\'', '')
    new_text = new_text.replace('.', '')
    new_text = new_text.replace(':', '')
    new_text = new_text.replace(')', '')
    new_text = new_text.replace('(', '')
    new_text = new_text.replace('[', '')
    new_text = new_text.replace(']', '')

    return new_text.strip()


def get_common_words(word_dict, threshold=0.9):
    """
    Returns a list of common words.

    :param word_dict: Word dict object containing both union and query-only dictionaries. See make_word_dict function.
    :param threshold: default: 0.9
    :return:
    """
    word_lengths = list(word_dict['All'].values())
    word_lengths.sort()
    common_words = [word for word in word_dict['All'].keys() if word_dict['All'][word] >= word_lengths[round(threshold * len(word_dict['All']))]]
    return [word for word in common_words if word not in word_dict['Query']]


def get_rare_words(word_dict, threshold=0.1):
    """
    Returns a list of rare words.

    :param word_dict: Word dict object containing both union and query-only dictionaries. See make_word_dict function.
    :param threshold: default: 0.1
    :return:
    """
    word_lengths = list(word_dict['All'].values())
    word_lengths.sort()
    rare_words = [word for word in word_dict['All'].keys() if word_dict['All'][word] <= word_lengths[round(threshold * len(word_dict['All']))]]
    return [word for word in rare_words if word not in word_dict['Query']]


def calc_bow(text, words):
    """
    Calculates Bag of Words (BOW) vector for a text and a list of words

    :param text: Text for which to calculate BOW vector.
    :param words: Array-like words
    :return: BOW vector which is same size as words vector.
    """
    tokens = text.split()
    return [sum([a == word for a in tokens]) for word in words]


def make_bow_dfs(df, word_dict, comment_column='Comment', query_column='Query'):
    """
    Makes BOW matrix in form of a pandas dataFrame

    :param df: Input dataframe (One row per comment-query pair)
    :param word_dict: Word dict object containing both union and query-only dictionaries. See make_word_dict function.
    :param queries: Array-like list of queries
    :param comment_column: Name of the column in the dataframe containing Comments
    :param query_column: Name of the column in the dataframe containing Queries
    :return: Two dataframes in a dict. One is called Comment and contains comment BOW vectors, while the other one is
    named Query and contains query vectors
    """

    unique_words = list(word_dict['All'].keys())
    bag_of_words = {pairID: calc_bow(df[df['PairID'] == pairID].iloc[0][comment_column], unique_words) for pairID in df['PairID'].unique()}

    bag_of_words_queries = {queryID: calc_bow(df[df['QueryID'] == queryID].iloc[0][query_column], unique_words) for queryID in df['QueryID'].unique()}

    df_BOW = pd.DataFrame.from_dict(bag_of_words, orient='index', columns=unique_words)
    df_BOWQ = pd.DataFrame.from_dict(bag_of_words_queries, orient='index', columns=unique_words)

    return {'Comment': df_BOW.copy(), 'Query': df_BOWQ.copy()}


def calc_tf(text, words):
    """
    Calculates Term Frequency (TF) vector for a text and a list of words.

    :param text: Text for which to calculate TF vector.
    :param words: Array-like words
    :return: TF vector which is same size as words vector.
    """
    num_words = len(text.split())
    return [c / num_words for c in calc_bow(text, words)]


def make_tf_dfs(bow_dfs):
    """
    Makes term frequency (TF) matrix in form of a pandas dataFrame.

    :param bow_dfs: Output of make_bow_dfs function.
    :return: Two dataframes in a dict. One is called Comment and contains comment TF vectors, while the other one is
    named Query and contains query TF vectors
    """
    tf = {}
    for pair_id in bow_dfs['Comment'].index:
        curr_sum = sum(bow_dfs['Comment'].loc[pair_id])
        tf[pair_id] = [c / curr_sum for c in bow_dfs['Comment'].loc[pair_id].values]

    tf_df = pd.DataFrame.from_dict(tf, orient='index', columns=bow_dfs['Comment'].columns)

    tfQ = {}
    for queryid in bow_dfs['Query'].index:
        curr_sum = sum(bow_dfs['Query'].loc[queryid])
        tfQ[queryid] = [c / curr_sum for c in bow_dfs['Query'].loc[queryid].values]

    tfQ_df = pd.DataFrame.from_dict(tfQ, orient='index', columns=bow_dfs['Query'].columns)

    return {'Comment': tf_df.copy(), 'Query': tfQ_df.copy()}


def calc_idf(df, bow_dfs):
    """
    Generates Inverse Data Frequency (IDF) vector.

    :param df: Input dataframe (One row per comment-query pair)
    :param bow_dfs: Output of make_bow_dfs function.
    :return: IDF vector
    """
    number_of_documents = df['PairID'].nunique()
    idf = [np.log(number_of_documents / (sum(bow_dfs['Comment'][word] > 0) + sum(bow_dfs['Query'][word] > 0))) for word in bow_dfs['Comment'].columns]
    return idf


def calc_tfidf(text, words, idf):
    """
        Calculates Term Frequency (TF) vector for a text and a list of words.

        :param text: Text for which to calculate TF vector.
        :param words: Array-like words
        :param idf: IDF vector
        :return: TF vector which is same size as words vector.
        """
    curr_tf = calc_tf(text, words)
    return [tf * idf for tf, idf in zip(curr_tf, idf)]


def make_tfidf_dfs(tf_dfs, idf):
    """
    Calculates Term Frequency - Inverse Data Frequency (TF-IDF) vector for a text and a list of words.

    :param tf_dfs: Output of make_tf_dfs
    :param idf: IDF vector. Produced by calc_idf
    :return: Two dataframes in a dict. One is called Comment and contains comment TFIDF vectors, while the other one is
    named Query and contains query TFIDF vectors
    """

    tfidf = tf_dfs['Comment'].copy().multiply(idf, axis=1)
    tfidfQ = tf_dfs['Query'].copy().multiply(idf, axis=1)

    return {'Comment': tfidf.copy(), 'Query': tfidfQ.copy()}


def cosine_sim(v1, v2):
    """
    Cosine similarity between two vectors. Vectors must be of same length

    :param v1: First vector
    :param v2: Second vector
    :return: Cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def make_base_features(df, inplace=False, comment_column='Comment', query_column='Query'):
    """
    Adds base features (text length, query length, common words (w, w/o repetition) to a dataframe

    :param df: Dataframe in which to add features.
    :param inplace: If True modify existing dataframe. If False, return a new dataframe.
    :param comment_column: Name of the column in the dataframe containing column text.
    :param query_column: Name of the column in the dataframe containing query text.
    :return: Either modifies input dataframe, or returns a new one containing both original columns as well as feature columns
    """

    if not inplace:
        temp_df = df.copy()
        temp_df['WordCount' + comment_column] = temp_df[comment_column].apply(lambda x: len(x.strip().split(' ')))
        temp_df['WordCount' + query_column] = temp_df[query_column].apply(lambda x: len(x.strip().split(' ')))
        temp_df['MutualUnique'] = temp_df.apply(lambda x: count_mutual_words(x[query_column], x[comment_column], remove_duplicates=True), axis=1)
        temp_df['MutualWithRepetition'] = temp_df.apply(lambda x: count_mutual_words(x[query_column], x[comment_column], remove_duplicates=False), axis=1)
        return temp_df
    else:
        df['WordCount' + comment_column] = df[comment_column].apply(lambda x: len(x.strip().split(' ')))
        df['WordCount' + query_column] = df[query_column].apply(lambda x: len(x.strip().split(' ')))
        df['MutualUnique'] = df.apply(lambda x: count_mutual_words(x[query_column], x[comment_column], remove_duplicates=True), axis=1)
        df['MutualWithRepetition'] = df.apply(lambda x: count_mutual_words(x[query_column], x[comment_column], remove_duplicates=False), axis=1)
    return


# TODO: ubaciti ngrame za n=5
def preprocess_text(df, output_file_name, lowercase=False, stem=False, remove_stop_words=False, freq_word_filtering=False, bigram_and_trigram=False,
                    tf_ponder=False, tf_idf_ponder=False):
    df_preprocess = df.copy()

    if lowercase:
        df_preprocess = preprocess_lowercase(df_preprocess)
    if stem:
        df_preprocess = preprocess_stem(df_preprocess)
    if freq_word_filtering:
        df_preprocess = preprocess_freq_word_filtering(df_preprocess)
    if bigram_and_trigram:
        df_preprocess = preprocess_bigrams_and_trigrams(df_preprocess)
    if tf_ponder:
        df_preprocess = preprocess_tf_ponder(df_preprocess)
    if tf_idf_ponder:
        df_preprocess = preprocess_tf_idf_ponder(df_preprocess)

    df_preprocess.to_csv(output_file_name, index=False)


def preprocess_lowercase(df):
    df_lowercase = normalize_to_lowercase(df, inplace=False)
    make_base_features(df_lowercase, inplace=True)

    # Prvo pravimo recnik:
    lowercase_wds = make_word_dict(df_lowercase)
    lowercase_bow = make_bow_dfs(df=df_lowercase, word_dict=lowercase_wds)

    # Dodajemo BOW kosinusnu sličnost kao dodatno obeležje
    df_lowercase['BOW'] = df_lowercase.apply(lambda x: cosine_sim(lowercase_bow['Comment'].loc[x['PairID']], lowercase_bow['Query'].loc[x['QueryID']]), axis=1)

    return df_lowercase


def preprocess_stem(df):
    df_stem = normalize_to_lowercase(df, inplace=False)
    apply_stem(df_stem, inplace=True)
    make_base_features(df_stem, inplace=True)

    stem_wds = make_word_dict(df_stem)
    stem_bow = make_bow_dfs(df=df_stem, word_dict=stem_wds)

    df_stem['BOW'] = df_stem.apply(lambda x: cosine_sim(stem_bow['Comment'].loc[x['PairID']], stem_bow['Query'].loc[x['QueryID']]), axis=1)

    return df_stem


def preprocess_freq_word_filtering(df_freq):
    # Izbacivanje interpunkcijskih znakova
    df_freq['Comment'] = df_freq['Comment'].apply(lambda x: remove_interpunction(x))
    df_freq['Query'] = df_freq['Query'].apply(lambda x: remove_interpunction(x))

    # Pravljanje liste čestih i retkih reči
    freq_words = make_word_dict(df_freq)
    common_words = get_common_words(freq_words)
    rare_words = get_rare_words(freq_words)

    # Frekvencijsko filtriranje tekstova i upita
    df_freq['Comment'] = df_freq['Comment'].apply(lambda x: remove_common_words(x, common_words))
    df_freq['Query'] = df_freq['Query'].apply(lambda x: remove_common_words(x, common_words))
    df_freq['Comment'] = df_freq['Comment'].apply(lambda x: remove_rare_words(x, rare_words))
    df_freq['Query'] = df_freq['Query'].apply(lambda x: remove_rare_words(x, rare_words))

    # Dodavanje osnovnih obelezja
    make_base_features(df_freq, inplace=True)

    # Racunanje BOW vektora
    # Pravimo novi rečnik koji neće sadržati filtrirane reči
    freq_words = make_word_dict(df_freq)
    freq_bow = make_bow_dfs(df=df_freq, word_dict=freq_words)

    # Dodajemo BOW kosinusnu sličnost kao dodatno obeležje
    df_freq['BOW'] = df_freq.apply(lambda x: cosine_sim(freq_bow['Comment'].loc[x['PairID']], freq_bow['Query'].loc[x['QueryID']]), axis=1)

    return df_freq


def preprocess_bigrams_and_trigrams(df_ngrams):
    # Dodavanje osnovnih obelezja
    make_base_features(df_ngrams, inplace=True)

    # Dodavanje broja zajednickih bigrama i trigrama
    df_ngrams['NCommonBigrams'] = df_ngrams.apply(lambda x: count_mutual_bigrams(x['Comment'], x['Query']), axis=1)

    df_ngrams['NCommonTrigrams'] = df_ngrams.apply(lambda x: count_mutual_trigrams(x['Comment'], x['Query']), axis=1)

    # Racunanje BOW vektora
    ngram_words = make_word_dict(df_ngrams)
    ngram_bow = make_bow_dfs(df=df_ngrams, word_dict=ngram_words)

    # Dodajemo BOW kosinusnu sličnost kao dodatno obeležje
    df_ngrams['BOW'] = df_ngrams.apply(lambda x: cosine_sim(ngram_bow['Comment'].loc[x['PairID']], ngram_bow['Query'].loc[x['QueryID']]), axis=1)

    df_ngrams.head(3)

    return df_ngrams


def preprocess_tf_ponder(df_tf):
    # Dodavanje osnovnih obelezja
    make_base_features(df_tf, inplace=True)

    # Racunanje TF vektora
    tf_words = make_word_dict(df_tf)
    tf_bow = make_bow_dfs(df=df_tf, word_dict=tf_words)
    tf_tf = make_tf_dfs(tf_bow)

    # Dodajemo TF kosinusnu sličnost kao dodatno obeležje
    df_tf['TF'] = df_tf.apply(lambda x: cosine_sim(tf_tf['Comment'].loc[x['PairID']], tf_tf['Query'].loc[x['QueryID']]), axis=1)

    return df_tf


def preprocess_tf_idf_ponder(df_tfidf):
    # Dodavanje osnovnih obelezja
    make_base_features(df_tfidf, inplace=True)

    # Racunanje TF vektora
    tfidf_words = make_word_dict(df_tfidf)
    tfidf_bow = make_bow_dfs(df=df_tfidf, word_dict=tfidf_words)
    tfidf_tf = make_tf_dfs(tfidf_bow)

    idf = calc_idf(df_tfidf, tfidf_bow)
    tfidf_tfidf = make_tfidf_dfs(tfidf_tf, idf)

    # Dodajemo TF kosinusnu sličnost kao dodatno obeležje
    df_tfidf['TFIDF'] = df_tfidf.apply(lambda x: cosine_sim(tfidf_tfidf['Comment'].loc[x['PairID']], tfidf_tfidf['Query'].loc[x['QueryID']]), axis=1)

    df_tfidf.head(3)

    return df_tfidf

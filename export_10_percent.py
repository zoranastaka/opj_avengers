import warnings

warnings.filterwarnings("ignore")

import pandas as pd

df = pd.read_csv('data/validation/for_validation_merged.csv')
df = df.iloc[:, 1:]

print(df.head())

lista_uzoraka = []
for i, row in df.iterrows():
    lista_uzoraka.append(['PHP', row['QueryID'], row['PairID'], row['QueryText'], row['CommentText'], row['SimilarityScore_1'],
                          row['SimilarityScore_2'], row['SimilarityScore_3'], row['SimilarityScore_4']])

df_10_percent = pd.DataFrame(lista_uzoraka, columns=['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore_1', 'SimilarityScore_2',
                                                 'SimilarityScore_3', 'SimilarityScore_4'])\
    .astype({'ProgrammingLanguageName': str, 'QueryID': str, 'PairID': str, 'QueryText': str, 'CommentText': str, 'SimilarityScore_1': str, 'SimilarityScore_2': str, 'SimilarityScore_3': str, 'SimilarityScore_4': str})


df_10_percent.to_csv('Dobar_format_10_percent.txt', sep='\t')

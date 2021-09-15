import pandas as pd

header_list = ['ProgrammingLanguageName', 'QueryID', 'PairID', 'QueryText', 'CommentText', 'SimilarityScore']
df_merged = pd.read_csv('../../merged_all.csv')
df_merged = df_merged.iloc[1:, :]

df_upiti = df_merged.copy().drop(['pair_id', 'comment', 'Komentar', 'code'], axis=1)
upiti = df_upiti.keys()

pairs_list = []
for i, row in df_merged.iterrows():
    query_id = 0
    for upit in upiti:
        pairs_list.append(['PHP', query_id, row['pair_id'], upit,  row['Komentar'], row[upit]])
        query_id += 1


print("Zoraana")
df_phase_2 = pd.DataFrame(pairs_list, columns=header_list)\
    .astype({'ProgrammingLanguageName': str, 'QueryID': int, 'PairID': str, 'QueryText': str, 'CommentText': str, 'SimilarityScore': str})

df_phase_2.to_csv('../../output_files/phase_2_data_annotation/Annotated_data_phase_2.txt', sep='\t', index=False)
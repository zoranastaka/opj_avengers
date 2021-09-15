import pandas as pd

header_list = ['ProgrammingLanguageName', 'RepoDescription', 'SourceDescription', 'PairID', 'CommentText']
df_overview = pd.read_csv('../../data/overview/overview.txt', delimiter='\t', names=header_list)
df_merged = pd.read_csv('../../merged_all.csv')
pair_ids_list = df_merged['pair_id'].tolist()


pairs_list = []
for i, row in df_overview.iterrows():
    if row['PairID'] in pair_ids_list:
        try:
            pairs_list.append([row['ProgrammingLanguageName'], row['RepoDescription'], row['SourceDescription'], row['PairID'], row['CommentText']])
            print("%r" % row['CommentText'])
        except:
            print(row)

print("Zoraana")
df_phase_1_2 = pd.DataFrame(pairs_list, columns=header_list)\
    .astype({'ProgrammingLanguageName': str, 'RepoDescription': str, 'SourceDescription': str, 'PairID': str, 'CommentText': str})

df_phase_1_2.to_csv('../../output_files/phase_1_collecting_data/phase_1_2/Overview_pairs_phase_1_2.txt', sep='\t')
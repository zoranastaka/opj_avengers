import pandas as pd
import shutil
import os.path

df_merged = pd.read_csv('../../merged_all.csv')

df_merged = df_merged.dropna()

print(df_merged.head())


cnt = 0

for index, row in df_merged.iterrows():
    target = r'F:/Master/OPJ/Projekat/projekat_GitHub/opj_avengers/output_files/phase_1_collecting_data/phase_1_1/filtered_files/' + row['pair_id'] + '.txt'
    originial = r'F:/Master/OPJ/Projekat/projekat_GitHub/opj_avengers/output_files/phase_1_collecting_data/phase_1_1/data_pairs_pt0_B_M/' + row['pair_id'] + '.txt'
    if not os.path.isfile(originial):
        originial = r'F:/Master/OPJ/Projekat/projekat_GitHub/opj_avengers/output_files/phase_1_collecting_data/phase_1_1/data_pairs_pt1_N_Y/' + row['pair_id'] + '.txt'

    if os.path.isfile(originial):
        pass
        shutil.copyfile(originial, target)
    else:
        print(row['pair_id'])
        cnt += 1

print(cnt)
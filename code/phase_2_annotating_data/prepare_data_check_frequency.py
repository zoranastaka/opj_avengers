import pandas as pd
import matplotlib.pyplot as plt

df_Boris = pd.read_csv('../../data/Boris/Boris_anotirano.csv')
boris_columns = [column.strip() for column in list(df_Boris.columns)]
df_Boris = pd.read_csv('../../data/Boris/Boris_anotirano_v2.csv', names=boris_columns, header=0)
df_Branislav = pd.read_csv('../../data/Branislav/annotation_matrix_Branislav.csv', names=boris_columns, header=0)
df_Zorana = pd.read_csv('../../data/Zorana/annotation_matrix_Zorana_konacno.csv', names=boris_columns, header=0)
df_Nikola = pd.read_csv('../../data/Nikola/anotaciona_konacna2_Nikola.csv', names=boris_columns, header=0)

df_merged = df_Boris.append(df_Branislav).append(df_Zorana).append(df_Nikola)

df_merged.to_csv('merged.csv')

df_merged.drop(columns='upit', inplace=True)
df_merged = df_merged.dropna()

df_merged.to_csv('../../merged_all.csv')

print("Ukupan broj parova je: {}".format(len(df_merged)))
df_merged.to_csv('./data/annotation_merged.csv', index=False)

# Ispitivanje učestanosti upita
descriptive_columns = ['pair_id', 'comment', 'Komentar', 'code']
queries = df_merged.drop(columns=descriptive_columns, inplace=False)
queries.head(1)

query_count = queries.fillna(0).astype(bool).sum(axis=0)
rare_queries = query_count[query_count < 1]
print(f"rare_queries has length of {len(rare_queries)}")
print("rare_queries")
print(rare_queries)

rare_queries.to_csv('Retki upiti.csv')
not_so_rare = rare_queries[rare_queries > 2]
not_so_rare.to_csv('Ne toliko retki upiti.csv')

query_count.to_csv('Upiti.csv')

for key in query_count.keys():
    query_count[key.strip().lower()] = query_count.pop(key)
query_count.pop('upit')

plt.figure(figsize=(5, 3), dpi=160)
plt.hist(query_count.values, bins = 25)
plt.xlabel('Broj pojavljivanja ne-nultih skorova')
plt.ylabel('Broj upita')
plt.title('Histogram broja upita anotiranih nenultim skorom sličnosti')
plt.grid(alpha=0.2)

plt.show()
















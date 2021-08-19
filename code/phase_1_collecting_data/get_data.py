import numpy as np
import pandas as pd

import scraper
from opj_avengers.code.phase_1_collecting_data import comment_parser

comment_parser.extract_comments_from_file('https://raw.githubusercontent.com/', '/jorgecasas/php-ml/raw/develop/src/DimensionReduction/PCA.php')

scraper.create_php_dict()
php_dict = scraper.create_php_dict(url='https://github.com/search?l=PHP&o=desc&q=php&s=stars&type=Repositories', limit=3000, max_files_per_repo=60, max_files_per_dir=10)
print(len(list(php_dict.keys())))


scraper.store_dict(data=php_dict, filename='new_php.json')
ml_dict_files = scraper.get_repo_files('https://github.com/jorgecasas/php-ml', limit=1000, dir_limit=500)
math_dict_files = scraper.get_repo_files('https://github.com/markrogoyski/math-php', limit=1000, dir_limit=700)
string_dict_files = scraper.get_repo_files('https://github.com/danielstjules/Stringy', limit=1000, dir_limit=1000)
image_string_files = scraper.get_repo_files('https://github.com/Intervention/image', limit=1000, dir_limit=500)
array_files = scraper.get_repo_files('https://github.com/spatie/array-functions', limit=100, dir_limit=10)
algorithms = scraper.get_repo_files('https://github.com/TheAlgorithms/PHP', limit=1000, dir_limit=500)

print(f"ml_dict_files length is: {len(ml_dict_files)}")
print(f"math_dict_files length is: {len(math_dict_files)}")
print(f"string_dict_files length is: {len(string_dict_files)}")
print(f"image_string_files length is: {len(image_string_files)}")
print(f"image_string_files length is: {len(image_string_files)}")
print(f"array_files length is: {len(array_files)}")
print(f"algorithms length is: {len(algorithms)}")

php_dict['https://github.com/jorgecasas/php-ml'] = ml_dict_files
php_dict['https://github.com/markrogoyski/math-php'] = math_dict_files
php_dict['https://github.com/danielstjules/Stringy'] = string_dict_files
php_dict['https://github.com/Intervention/image'] = image_string_files
php_dict['https://github.com/spatie/array-functions'] = array_files

new_dict = {}
new_dict['https://github.com/thephpleague/csv'] = scraper.get_repo_files('https://github.com/thephpleague/csv', limit=100, dir_limit=100)
new_dict['https://github.com/ThingEngineer/PHP-MySQLi-Database-Class'] = scraper.get_repo_files('https://github.com/ThingEngineer/PHP-MySQLi-Database-Class', limit=100, dir_limit=100)
new_dict['https://github.com/ircmaxell/RandomLib'] = scraper.get_repo_files('https://github.com/ircmaxell/RandomLib', limit=200, dir_limit=100)
new_dict['https://github.com/denissimon/prediction-builder'] = scraper.get_repo_files('https://github.com/denissimon/prediction-builder')
new_dict['https://github.com/RubixML/ML/tree/master/src/Clusterers'] = scraper.get_repo_files('https://github.com/RubixML/ML/tree/master/src/Clusterers')
new_dict['https://github.com/googleapis/google-cloud-php/tree/master/AutoMl/src/V1beta1/ClassificationEvaluationMetrics'] = scraper.get_repo_files('https://github.com/googleapis/google-cloud-php/tree/master/AutoMl/src/V1beta1/ClassificationEvaluationMetrics')
new_dict['https://github.com/RubixML/ML/tree/master/src/Regressors'] = scraper.get_repo_files('https://github.com/RubixML/ML/tree/master/src/Regressors')

scraper.store_dict(new_dict, 'new_files_21.json')

counter = 0
for key in new_dict:
    counter += len(new_dict[key])
print(counter)

php_dict['https://github.com/TheAlgorithms/PHP'] = algorithms
php_dict = scraper.read_json('new_php.json')

dict_array = [{} for i in range(12)]

for i, key in enumerate(php_dict):
    dict_array[i % 12][key] = php_dict[key]

for i in range(12):
    scraper.store_dict(data=dict_array[i], filename='php_{}.json'.format(str(i)))


print(f"php_dict length is: {len(list(php_dict.keys()))}")

comment_parser.extract_comments('php_5.json', verbose=False)


for j in range(7, 8):
    print("Processing php_{}.json".format(j))
    comment_parser.extract_comments('php_{}.json'.format(j), verbose=True)

dicts_of_interest = [
    'https://github.com/jorgecasas/php-ml',
    'https://github.com/markrogoyski/math-php',
    'https://github.com/danielstjules/Stringy',
    'https://github.com/Intervention/image',
    'https://github.com/spatie/array-functions'
]

for i in range(12):
    temp_dict = scraper.read_json('php_{}.json'.format(i))
    for d in dicts_of_interest:
        if d in temp_dict:
            print(i)
            print(d)

a = scraper.read_json('php_0.json')

a.keys()

data1 = pd.read_csv('/Users/boris_majic/Documents/ETF/OPJ/opj_avengers/data/annotation_matrix.csv')

data1 = pd.read_csv('data/annotation_matrix_pt1.csv')

# Remove NA comments
data1 = data1[~data1.comment.isna()]

# Remove decorator comments
data1 = data1[~data1.comment.str.startswith(" @")]

data2 = pd.read_csv('data/annotation_matrix.csv', low_memory=False)

# Remove NA comments
data2 = data2[~data2.comment.isna()]

# Remove decorator comments
data2 = data2[~data2.comment.str.startswith(" @")]

data_all = data2.append(data1, ignore_index = True)

print(f"data_all length is {len(data_all)}")

ixs = np.arange(data_all.shape[0])
np.random.shuffle(ixs)
n_split = 4
# np.split cannot work when there is no equal division
# so we need to find out the split points ourself
# we need (n_split-1) split points
split_points = [i * data_all.shape[0] // n_split for i in range(1, n_split)]
# use these indices to select the part we want
data_part = []
for ix in np.split(ixs, split_points):
    data_part.append(data_all.iloc[ix].copy())
#     print(data1.iloc[ix])

for i in range(len(data_part)):
    data_part[i].insert(2, "Komentar", "")
    data_part[i].sort_values("pair_id", inplace=True)

data_part[0].to_csv("data/Boris/annotation_matrix_Boris.csv", index=False)
data_part[1].to_csv("data/Zorana/annotation_matrix_Zorana.csv", index=False)
data_part[2].to_csv("data/Branislav/annotation_matrix_Branislav.csv", index=False)
data_part[3].to_csv("data/Nikola/annotation_matrix_Nikola.csv", index=False)

print(data_part[3].head())
print("data part starts with @")
print(data_part[1][data_part[1].comment.str.startswith("@")])

data_part[1] = data_part[1][~data_part[1].comment.isna()]
data_part[1] = data_part[1][~data_part[1].comment.str.startswith(" @")]
data_part[1].insert(2, "Komentar na sprskom", "")
print(data_part[1])

srpski_dict = {
    'https://github.com/turshija/KGB-Game-Panel-Restarter': '/turshija/KGB-Game-Panel-Restarter/raw/master/assets/classes/restarter.class.php',
    'https://github.com/turshija/JavaScript-Multiplayer-Game': '/turshija/JavaScript-Multiplayer-Game/raw/master/index.php',
    'https://github.com/mateastanisic/searchme': '/mateastanisic/searchme/raw/master/app/init.php',
    'https://github.com/mateastanisic/searchme': '/mateastanisic/searchme/raw/master/app/controller_base.class.php',
}

for key in srpski_dict:
    comment_parser.extract_comments_from_file(key, srpski_dict[key])

srpski_dict = {}
srpski_dict['https://github.com/turshija/KGB-Game-Panel-Restarter'] = scraper.get_repo_files('https://github.com/turshija/KGB-Game-Panel-Restarter')
srpski_dict['https://github.com/turshija/JavaScript-Multiplayer-Game'] = scraper.get_repo_files('https://github.com/turshija/JavaScript-Multiplayer-Game')
srpski_dict['https://github.com/mateastanisic/searchme'] = scraper.get_repo_files('https://github.com/mateastanisic/searchme')
srpski_dict['https://github.com/besomuk/JMBG'] = scraper.get_repo_files('https://github.com/besomuk/JMBG')
srpski_dict['https://github.com/mateastanisic/studentplus'] = scraper.get_repo_files('https://github.com/mateastanisic/studentplus')
srpski_dict['https://github.com/laxsrbija/piro'] = scraper.get_repo_files('https://github.com/laxsrbija/piro')
srpski_dict['https://github.com/NevenaDarijevic/AplikacijaPHP'] = scraper.get_repo_files('https://github.com/NevenaDarijevic/AplikacijaPHP')
srpski_dict['https://github.com/DamirLuketic/to_do_app'] = scraper.get_repo_files('https://github.com/DamirLuketic/to_do_app')
# srpski_dict[] = scraper.get_repo_files()
# srpski_dict[] = scraper.get_repo_files()
# srpski_dict[] = scraper.get_repo_files()
# srpski_dict[] = scraper.get_repo_files()

scraper.store_dict(srpski_dict, filename='srpski.json')
comment_parser.extract_comments('new_files_21.json')

import glob, os
os.chdir("./jsons")
files = glob.glob("*.json")

all_repos = []
for file in files:
    all_repos += list(scraper.read_json(file))

with open('list_of_repos.txt', 'w') as file:
    file.writelines(all_repos)

df = pd.read_csv('./data/annotation_merged.csv').drop(columns=['Unnamed: 0'])
df.head()

dict_1 = scraper.read_json('srpski.json')
dict_2 = scraper.read_json('php_files.json')
dict_3 = scraper.read_json('new_php.json')
dict_4 = scraper.read_json('new_files_21.json')


dicts = [dict_1, dict_2, dict_3, dict_4]

all_data = {}
for d in dicts:
    for key in d:
        all_data[key] = d[key].copy()

with open('data/overview/overview.txt', 'w') as file:
    for index, row in df.iterrows():
        curr_id = row.pair_id
        id_parts = curr_id.split('_')
        # print(curr_id)
        curr_repo = 'https://github.com/' + id_parts[0] + '/' + id_parts[1]

        if len(id_parts) == 4:
            file_basename = id_parts[2]
        else:
            file_basename = '_'.join(id_parts[2:-1])
        try:
            for f in all_data[curr_repo]:
                if f.endswith(file_basename + '.php'):
                    curr_filepath = f
                    break
            else:
                print('File not found: {}'.format(curr_id))
        except KeyError:
            curr_repo = 'https://github.com/' + id_parts[0] + '/' + id_parts[1] + '_' + id_parts[2]
            if len(id_parts) == 5:
                file_basename = id_parts[3]
            else:
                file_basename = '_'.join(id_parts[3:-1])
            try:
                for f in all_data[curr_repo]:
                    if f.endswith(file_basename + '.php'):
                        curr_filepath = f
                        break
                else:
                    print('File not found: {}'.format(curr_id))
            except KeyError:
                # Ovaj repo je dat na drugi nacin
                for key in all_data:
                    if key.endswith(id_parts[0] + '/' + id_parts[1]):
                        curr_repo = key
                        break

                #                 print(curr_id)
                if len(id_parts) == 4:
                    file_basename = id_parts[2]
                else:
                    file_basename = '_'.join(id_parts[2:-1])

                try:
                    for f in all_data[curr_repo]:
                        if f.endswith(file_basename + '.php'):
                            curr_filepath = f
                            break
                    else:
                        print('File not found: {}'.format(curr_id))
                except KeyError:
                    print(curr_id)
                    continue

        curr_line = 'PHP\t' + curr_repo + '\tgithub.com/' + curr_filepath + '\t' + curr_id + '\t' + row.comment + '\n'
        file.write(curr_line)



































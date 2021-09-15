import numpy as np
import pandas as pd
from nltk import agreement
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv('data/validation/for_validation_merged.csv')

annotator_1 = df['SimilarityScore_1'][:15404]
annotator_2 = df['SimilarityScore_2'][:15404]
annotator_3 = df['SimilarityScore_3'][:15404]
annotator_4 = df['SimilarityScore_4'][:15404]

# summarize
print('annotator_1: mean=%.3f stdv=%.3f' % (np.mean(annotator_1), np.std(annotator_1)))
print('annotator_2: mean=%.3f stdv=%.3f' % (np.mean(annotator_2), np.std(annotator_2)))
print('annotator_3: mean=%.3f stdv=%.3f' % (np.mean(annotator_3), np.std(annotator_3)))
print('annotator_4: mean=%.3f stdv=%.3f' % (np.mean(annotator_4), np.std(annotator_4)))

# sm_1_2 = difflib.SequenceMatcher(None, annotator_1, annotator_2)
# print("Similarity between annotator 1 and 2: " + str(sm_1_2.ratio()))

# sm_1_2 = difflib.SequenceMatcher(None, annotator_2, annotator_1)
# print("Similarity between annotator 2 and 1: " + str(sm_1_2.ratio()))

print(annotator_1.isnull().values.any())
print(annotator_2.isnull().values.any())
print(annotator_3.isnull().values.any())
print(annotator_4.isnull().values.any())

print("PEARSON'S correlation")
# PEARSON'S correlation
corr, _ = pearsonr(annotator_1, annotator_2)
print('Pearsons correlation 1 and 2: %.3f' % corr)

corr, _ = pearsonr(annotator_1, annotator_3)
print('Pearsons correlation 1 and 3: %.3f' % corr)

corr, _ = pearsonr(annotator_1, annotator_4)
print('Pearsons correlation 1 and 4: %.3f' % corr)

corr, _ = pearsonr(annotator_2, annotator_3)
print('Pearsons correlation 2 and 3: %.3f' % corr)

corr, _ = pearsonr(annotator_2, annotator_4)
print('Pearsons correlation 2 and 4: %.3f' % corr)

corr, _ = pearsonr(annotator_3, annotator_4)
print('Pearsons correlation 3 and 4: %.3f' % corr)

print("SPEARMANS'S correlation")
# SPEARMANS'S correlation
corr, _ = spearmanr(annotator_1, annotator_2)
print('Spearmans correlation 1 and 2: %.3f' % corr)

corr, _ = spearmanr(annotator_1, annotator_3)
print('Spearmans correlation 1 and 3: %.3f' % corr)

corr, _ = spearmanr(annotator_1, annotator_4)
print('Spearmans correlation 1 and 4: %.3f' % corr)

corr, _ = spearmanr(annotator_2, annotator_3)
print('Spearmans correlation 2 and 3: %.3f' % corr)

corr, _ = spearmanr(annotator_2, annotator_4)
print('Spearmans correlation 2 and 4: %.3f' % corr)

corr, _ = spearmanr(annotator_3, annotator_4)
print('Spearmans correlation 3 and 4: %.3f' % corr)

print("Cohen’s kappa coefficient")
# Cohen’s kappa coefficient
cohen_score = cohen_kappa_score(annotator_1, annotator_2)
print('Cohen kappa score 1 and 2: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_1, annotator_3)
print('Cohen kappa score 1 and 3: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_1, annotator_4)
print('Cohen kappa score 1 and 4: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_2, annotator_3)
print('Cohen kappa score 2 and 3: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_2, annotator_4)
print('Cohen kappa score 2 and 4: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_3, annotator_4)
print('Cohen kappa score 3 and 4: %.3f' % cohen_score)


taskdata = [[0, str(i), str(annotator_1[i])] for i in range(0, len(annotator_1))] + [[1, str(i), str(annotator_2[i])] for i in range(0, len(annotator_2))] + [[2, str(i), str(annotator_3[i])] for i in range(0, len(annotator_3))] + [[3, str(i), str(annotator_4[i])] for i in range(0, len(annotator_4))]

ratingtask = agreement.AnnotationTask(data=taskdata)
print("Kappa " + str(ratingtask.kappa()))
print("Fleiss " + str(ratingtask.multi_kappa()))
print("Alpha " + str(ratingtask.alpha()))
print("Scotts " + str(ratingtask.pi()))


df = pd.read_csv('data/validation/for_validation_merged_none_zero.csv')

annotator_1 = df['SimilarityScore_1'][:92]
annotator_2 = df['SimilarityScore_2'][:92]
annotator_3 = df['SimilarityScore_3'][:92]
annotator_4 = df['SimilarityScore_4'][:92]

# summarize
print('annotator_1: mean=%.3f stdv=%.3f' % (np.mean(annotator_1), np.std(annotator_1)))
print('annotator_2: mean=%.3f stdv=%.3f' % (np.mean(annotator_2), np.std(annotator_2)))
print('annotator_3: mean=%.3f stdv=%.3f' % (np.mean(annotator_3), np.std(annotator_3)))
print('annotator_4: mean=%.3f stdv=%.3f' % (np.mean(annotator_4), np.std(annotator_4)))


print(annotator_1.isnull().values.any())
print(annotator_2.isnull().values.any())
print(annotator_3.isnull().values.any())
print(annotator_4.isnull().values.any())

print("PEARSON'S correlation")
# PEARSON'S correlation
corr, _ = pearsonr(annotator_1, annotator_2)
print('Pearsons correlation 1 and 2: %.3f' % corr)

corr, _ = pearsonr(annotator_1, annotator_3)
print('Pearsons correlation 1 and 3: %.3f' % corr)

corr, _ = pearsonr(annotator_1, annotator_4)
print('Pearsons correlation 1 and 4: %.3f' % corr)

corr, _ = pearsonr(annotator_2, annotator_3)
print('Pearsons correlation 2 and 3: %.3f' % corr)

corr, _ = pearsonr(annotator_2, annotator_4)
print('Pearsons correlation 2 and 4: %.3f' % corr)

corr, _ = pearsonr(annotator_3, annotator_4)
print('Pearsons correlation 3 and 4: %.3f' % corr)

print("SPEARMANS'S correlation")
# SPEARMANS'S correlation
corr, _ = spearmanr(annotator_1, annotator_2)
print('Spearmans correlation 1 and 2: %.3f' % corr)

corr, _ = spearmanr(annotator_1, annotator_3)
print('Spearmans correlation 1 and 3: %.3f' % corr)

corr, _ = spearmanr(annotator_1, annotator_4)
print('Spearmans correlation 1 and 4: %.3f' % corr)

corr, _ = spearmanr(annotator_2, annotator_3)
print('Spearmans correlation 2 and 3: %.3f' % corr)

corr, _ = spearmanr(annotator_2, annotator_4)
print('Spearmans correlation 2 and 4: %.3f' % corr)

corr, _ = spearmanr(annotator_3, annotator_4)
print('Spearmans correlation 3 and 4: %.3f' % corr)

print("Cohen’s kappa coefficient")
# Cohen’s kappa coefficient
cohen_score = cohen_kappa_score(annotator_1, annotator_2)
print('Cohen kappa score 1 and 2: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_1, annotator_3)
print('Cohen kappa score 1 and 3: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_1, annotator_4)
print('Cohen kappa score 1 and 4: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_2, annotator_3)
print('Cohen kappa score 2 and 3: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_2, annotator_4)
print('Cohen kappa score 2 and 4: %.3f' % cohen_score)

cohen_score = cohen_kappa_score(annotator_3, annotator_4)
print('Cohen kappa score 3 and 4: %.3f' % cohen_score)


taskdata = [[0, str(i), str(annotator_1[i])] for i in range(0, len(annotator_1))] + [[1, str(i), str(annotator_2[i])] for i in range(0, len(annotator_2))] + [[2, str(i), str(annotator_3[i])] for i in range(0, len(annotator_3))] + [[3, str(i), str(annotator_4[i])] for i in range(0, len(annotator_4))]

ratingtask = agreement.AnnotationTask(data=taskdata)
print("Kappa " + str(ratingtask.kappa()))
print("Fleiss " + str(ratingtask.multi_kappa()))
print("Alpha " + str(ratingtask.alpha()))
print("Scotts " + str(ratingtask.pi()))
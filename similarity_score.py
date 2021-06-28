import pandas as pd
import difflib

df = pd.read_csv('data/validation/for_validation_merged.csv')

annotator_1 = df['SimilarityScore_1']
annotator_2 = df['SimilarityScore_2']
annotator_3 = df['SimilarityScore_3']
annotator_4 = df['SimilarityScore_4']

sm_1_2 = difflib.SequenceMatcher(None, annotator_1, annotator_2)
print("Similarity between annotator 1 and 2: " + str(sm_1_2.ratio()))

sm_1_2 = difflib.SequenceMatcher(None, annotator_2, annotator_1)
print("Similarity between annotator 1 and 2: " + str(sm_1_2.ratio()))

sm_1_3 = difflib.SequenceMatcher(None, annotator_3, annotator_1)
print("Similarity between annotator 1 and 3: " + str(sm_1_3.ratio()))

sm_1_4 = difflib.SequenceMatcher(None, annotator_1, annotator_4)
print("Similarity between annotator 1 and 4: " + str(sm_1_4.ratio()))

sm_3_2 = difflib.SequenceMatcher(None, annotator_3, annotator_2)
print("Similarity between annotator 3 and 2: " + str(sm_3_2.ratio()))

sm_2_4 = difflib.SequenceMatcher(None, annotator_2, annotator_4)
print("Similarity between annotator 2 and 4: " + str(sm_2_4.ratio()))

sm_3_4 = difflib.SequenceMatcher(None, annotator_3, annotator_4)
print("Similarity between annotator 3 and 4: " + str(sm_3_4.ratio()))

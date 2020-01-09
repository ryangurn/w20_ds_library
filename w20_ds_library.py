import pandas as pd
import string
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')

def compile_vector(target_row):
  ret_vec = []
  ret_vec.append(target_row['Length'])
  for l in letters:
    ret_vec.append(target_row[l])

  return ret_vec

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"

  dist = 0.0
  for i in range(0,len(vect1)):
    dist += (vect1[i] - vect2[i])**2

  return np.sqrt(dist)
  #rest of your code below

def ordered_distances(target_vector:list, crowd_table:dframe, dfunc:Callable) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'

  #your code goes here  
  distance_list = []

  for i in range(0, len(crowd_table)):
    row = crowd_table.iloc[i]
    vect = compile_vector(row)

    distance_list.append((i, dfunc(target_vector, vect)))

  distance_list = sorted(distance_list, key = lambda x: x[1])
  return distance_list

def knn(target_vector:list, crowd_table:dframe, k:int, dfunc:Callable) -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert crowd_table['Survived'].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'

  sort = ordered_distances(target_vector, crowd_table, dfunc)
  #your code goes here
  top_k = [i for i,d in sort[:k]]
  opinions = [crowd_table.loc[i, 'Survived'] for i in top_k]
  winner = 1 if opinions.count(1) > opinions.count(0) else 0
  return winner

def knn_tester(test_table, crowd_table, k, dfunc:Callable) -> dict:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert crowd_table['Survived'].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  
  overall_sum = 0
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  for target_vector in range(0, len(test_table)):
    vect = compile_vector(test_table.iloc[target_vector])
    predicted = knn(vect, crowd_table, k, dfunc)
    actual = test_table.iloc[target_vector]['Survived']
    if predicted == actual == 1:
      true_positive += 1
    elif predicted == actual == 0:
      true_negative += 1
    if predicted != actual and predicted == 1:
      false_positive += 1
    elif predicted != actual and predicted == 0:
      false_negative += 1
    overall_sum += 1
  return {(0,0): true_negative, (0,1): false_negative, (1,0): false_positive, (1,1): true_positive}

def cm_accuracy(confusion_dictionary: dict) -> float:
  assert isinstance(confusion_dictionary, dict), f'confusion_dictionary not a dictionary but instead a {type(confusion_dictionary)}'
  
  tp = confusion_dictionary[(1,1)]
  fp = confusion_dictionary[(1,0)]
  fn = confusion_dictionary[(0,1)]
  tn = confusion_dictionary[(0,0)]
  
  return (tp+tn)/(tp+fp+fn+tn)

def cm_f1(confusion_dictionary: dict) -> float:
  assert isinstance(confusion_dictionary, dict), f'confusion_dictionary not a dictionary but instead a {type(confusion_dictionary)}'
  
  tp = confusion_dictionary[(1,1)]
  fp = confusion_dictionary[(1,0)]
  fn = confusion_dictionary[(0,1)]
  tn = confusion_dictionary[(0,0)]
  
  recall = tp/(tp+fn) if (tp+fn) != 0 else 0  #Heuristic
  precision = tp/(tp+fp) if (tp+fp) != 0 else 0  #Heuristic
  
  recall_div = 1/recall if recall != 0 else 0  #Heuristic
  precision_div = 1/precision if precision != 0 else 0  #Heuristic
  
  f1 = 2/(recall_div+precision_div) if (recall_div+precision_div) != 0 else 0  #heuristic
  
  return f1


def hello_ds():
    print("Big hello to you")

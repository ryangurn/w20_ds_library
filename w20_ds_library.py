import pandas as pd
import string
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')


#############
#  module1  #
#############

def compile_vector(row, columns):
  nt = row.drop(columns)
  return nt.to_list()

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"

  dist = 0.0
  for i in range(0,len(vect1)):
    dist += (vect1[i] - vect2[i])**2

  return np.sqrt(dist)

def ordered_distances(target_vector:list, crowd_table:dframe, answer_column:str, dfunc:Callable) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'

  crowd = crowd_table.drop(answer_column, axis=1) # drop the answer column from the results (copy)
  dist = [(i, dfunc(target_vector, v.tolist())) for i, v in crowd.iterrows()] # condensing my for loop down
  return sorted(dist, key=lambda x: x[1]) # sort with regard to position 1... https://stackoverflow.com/questions/29781862/how-does-this-example-of-a-lambda-function-work

def knn(target_vector:list, crowd_table:dframe, answer_column:str, k:int, dfunc:Callable) -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'

  sort = ordered_distances(target_vector, crowd_table, answer_column, dfunc)
  #your code goes here
  top_k = [i for i,d in sort[:k]]
  opinions = [crowd_table.loc[i, answer_column] for i in top_k]
  winner = 1 if opinions.count(1) > opinions.count(0) else 0
  return winner

def knn_tester(test_table, crowd_table, answer_column, k, dfunc:Callable) -> dict:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  
  overall_sum = 0
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  for target_vector in range(0, len(test_table)):
    vect = compile_vector(test_table.iloc[target_vector], [answer_column])
    predicted = knn(vect, crowd_table, answer_column, k, dfunc)
    actual = test_table.iloc[target_vector][answer_column]
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


#############
#  module2  #
#############

def dot_product(vect1: list, vect2: list) -> int:
  return sum(x*y for x,y in zip(vect1, vect2))

def square_root(input: int) -> float:
  last_guess = input / 2.0
  while True:
    guess = (last_guess + input / last_guess) / 2
    if abs(guess - last_guess) < 1*10**(-32):
      return guess
    last_guess = guess

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  

  #your code here
  return dot_product(vect1, vect2) / (square_root(dot_product(vect1, vect1)) * square_root(dot_product(vect2, vect2)))

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"

  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result

def filter_by_column_value(df, col, value):
  filtered_table = df.loc[df[col] == value]
  return filtered_table

#############
#  module3  #
#############

def bayes(evidence:set, evidence_bag:dict, training_table:dframe) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  labels = training_table.label.to_list()
  class_amt = len(set(labels))
  assert len(list(evidence_bag.values())[0]) == class_amt, f'The number of values in evidence_bag does not match the number of unique classes ({class_amt}) in labels.'

  counter = []
  probabilities = []
  for i in range(class_amt):
    ct = labels.count(i)
    counter.append(ct)
    probabilities.append(ct/len(labels))

  res = []
  for j in range(class_amt):
    numerator = 1
    for i in evidence:
      all_values = evidence_bag[i]
      the_value = (all_values[j]/counter[j])
      numerator *= the_value
    res.append(numerator * probabilities[j])

  return tuple(res)

def char_set_builder(text:str) -> list:
  the28 = set(text).intersection(set('abcdefghijklmnopqrstuvwxyz!#'))
  return list(the28)

def bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'

  res = []
  for i,j in testing_table.iterrows():
    r = j['text'] # raw text
    p = set(parser(r)) # parse it
    t = bayes(p, evidence_bag, training_table) # bayes it
    res.append(t) # append the tuple
  return res

def hello_ds():
    print("Big hello to you")

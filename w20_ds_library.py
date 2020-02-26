import pandas as pd
import string
import re
import numpy as np 
from typing import TypeVar, Callable
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
dframe = TypeVar('pd.core.frame.DataFrame')
narray = TypeVar('np.ndarray')
spnlp = TypeVar('spacy.lang.en.English')


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
  try:
    return dot_product(vect1, vect2) / (square_root(dot_product(vect1, vect1)) * square_root(dot_product(vect2, vect2)))
  except ZeroDivisionError:
    return 0.0

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

#############
#  module4  #
#############

def get_clean_words(stopwords:list, raw_sentence:str) -> list:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a str but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return cleaned

def build_word_bag(stopwords:list, training_table:dframe) -> dict:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  retDict = {}
  outputLabels = [[1,0,0], [0,1,0], [0,0,1]]
  for key, value in training_table.iterrows():
    text = value['text']
    words = set(get_clean_words(stopwords, text)) # count only first occurance to determine feature words
    label =  value['label']
    for word in words:
        if word in retDict: 
            retDict[word][label] += 1 # count the number of occurances
        else:
            retDict[word] = list(outputLabels[label])  #must be a list to get a copy
  return retDict

def robust_bayes(evidence:set, evidence_bag:dict, training_table:dframe, laplace:float=1.0) -> tuple:
	assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
	assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
	assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
	assert 'label' in training_table, f'label column is not found in training_table'
	assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

	labels = training_table.label.to_list()
	classes = len(set(labels))
	assert len(list(evidence_bag.values())[0]) == classes, f'Values in evidence_bag do not match number of unique classes ({classes}) in labels.'

	counts = []
	probs = []
	for i in range(classes):
		ct = labels.count(i)
		counts.append(ct)
		probs.append(ct/len(labels))

	#now have counts and probs for all classes

	results = []
	for clss in range(classes):
		num = 1
		for ei in evidence:
			if ei not in evidence_bag:
				the_value =  1/(counts[clss] + len(evidence_bag) + laplace)
			else:
				all_values = evidence_bag[ei]
				the_value = ((all_values[clss]+laplace)/(counts[clss] + len(evidence_bag) + laplace)) 
			num *= the_value
		results.append(max(num * probs[clss], 2.2250738585072014e-308))

	return tuple(results)


def robust_bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'

  res = []
  for i,row in testing_table.iterrows():
    txt = row['text']
    sset = set(parser(txt))
    tup = robust_bayes(sset, evidence_bag, training_table)
    res.append(tup)
  return res

def curry_x(old_func,x):
  #the body of curry_x is the closure. It contains the value of old_func and x.

  def inner(y): return old_func(x,y)  #define a new function that uses x from closure and passes in y.

  return inner  #will return a function of 1 argument. But that function will carry the closure with it.

def curry_x(old_func,x):
  #this is the closure. It contains the value of x.

  def inner(y): return old_func(x,y)  #define a new function that uses x from closure and passes in y.

  return inner  #will return a function of 1 argument. But that function will carry the closure with it.

swords = stopwords.words('english')
swords.sort()
word_parser = curry_x(get_clean_words, swords)

def cmat_accuracy(cmat:list) -> float:
  assert isinstance(cmat, list), f'cmat not a list but instead a {type(cmat)}'
  assert all(len(row) == len(cmat) for row in cmat), f'cmat not a square matrix'

  total = 0
  correct = 0
  for i in range(0,3):
    for j in range(0,3):
      if i == j:
        correct += cmat[i][j]
      total += cmat[i][j]
  return correct/total

#############
#  module8  #
#############

def ordered_animals(target_vector, table):
  #distance_list = [(row.name,euclidean_distance(target_vector, row.tolist())) for i,row in table.iterrows()]  #for list comprehension fans
  distance_list = []
  for i,row in table.iterrows():
    a_vector = row.tolist()
    d = euclidean_distance(target_vector, a_vector)
    distance_list.append((row.name,d))
  return sorted(distance_list, key=lambda pair: pair[1], reverse=False)

def hex_to_int(s:str) -> tuple:
  assert isinstance(s, str)
  assert s[0] == '#'
  assert len(s) == 7

  s = s.lstrip("#")
  return int(s[:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def fast_euclidean_distance(x:narray, y:narray) -> float:
  assert isinstance(x, np.ndarray), f"x must be a np array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, np.ndarray), f"y must be a np array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.linalg.norm(x-y)

def subtractv(x:narray, y:narray) -> narray:
  assert isinstance(x, np.ndarray), f"x must be a np array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, np.ndarray), f"y must be a np array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"

  return np.subtract(x,y)

def addv(x:narray, y:narray) -> narray:
  assert isinstance(x, np.ndarray), f"x must be a np array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, np.ndarray), f"y must be a np array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.add(x,y)

def meanv_slow(coords):
    # assumes every item in coords has same length as item 0
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean

def meanv(matrix: narray) -> narray:
  assert isinstance(matrix, np.ndarray), f"matrix must be a np array but instead is {type(matrix)}"
  assert len(matrix.shape) == 2, f"matrix must be a 2d array but instead is {len(matrix.shape)}d"

  return matrix.mean(axis=0)

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, np.ndarray), f"v1 must be a np array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, np.ndarray), f"v2 must be a np array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'

  if norm(v1) == 0 or norm(v2) == 0 or (norm(v1)*norm(v2)) == 0:
    return 0.0

  return np.dot(v1,v2)/(norm(v1)*norm(v2))

def dict_ordered_distances(space:dict, coord:narray) -> list:
  assert isinstance(space, dict), f"space must be a dictionary but instead a {type(space)}"
  assert isinstance(list(space.values())[0], np.ndarray), f"space must have np arrays as values but instead has {type(space.values()[0])}"
  assert isinstance(coord, np.ndarray), f"coord must be a np array but instead is {type(cord)}"
  assert len(list(space.values())[0]) == len(coord), f"space values must be same length as coord"
  assert len(coord) == 3, "coord must be a triple"

  return sorted([(key, fast_euclidean_distance(value, coord)) for key, value in space.items()], key = lambda x: x[1])

def vec(nlp:spnlp, s:str) -> narray:
    return nlp.vocab[s].vector

def sent2vec(nlp:spnlp, s: str) -> narray:
  ret_arr = []
  for i in nlp(s.lower()):
    ret_arr.append(vec(nlp, i.text))
  return meanv(np.matrix(ret_arr))

def spacy_closest_sent(nlp:spnlp, space:list, input_str:str, n:int=10):
  assert isinstance(space, list)
  assert all([isinstance(sp, spacy.tokens.span.Span) for sp in space])

  input_vec = sent2vec(nlp, input_str)
  return sorted(space,
                key=lambda x: fast_cosine(np.mean([w.vector for w in x], axis=0), input_vec),
                reverse=True)[:n]

def hello_ds():
    print("Big hello to you")

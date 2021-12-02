import numpy as np
import pandas as pd
import re

def transition_matrix(transitions):
    df = pd.DataFrame(transitions)

    # create a new column with data shifted one space
    df['shift'] = df[0].shift(-1)

    # add a count column (for group by function)
    df['count'] = 1

    # groupby and then unstack, fill the zeros
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)

    # normalise by occurences and save values to get transition matrix
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    return trans_mat

def make_pairs(data):
    for i in range(len(data)-1):
        yield (data[i], data[i+1])

strPath = "constitution.txt"
f = open(strPath)
strText = f.read()
strText = re.sub('([.,!?():;])', r' \1 ', strText)
strText = re.sub('\s{2,}', ' ', strText)
listText = strText.split()
m = transition_matrix(listText)
uniqueText = np.unique(listText)

pairs = make_pairs(listText)
wordDict = {}
for word_1, word_2 in pairs:
    if word_1 in wordDict.keys():
        wordDict[word_1].append(word_2)
    else:
        wordDict[word_1] = [word_2]


def generate_consitution_words(n):
    wordList = []
    first_word = "word"
    while first_word.islower():
        first_word = np.random.choice(listText)
        if first_word in ['!', ',', '.', '?', ':', ';']:
            first_word = np.random.choice(listText)
    wordList.append(first_word)

    for i in range(n):
        probabilities = []
        uniqueDict = np.unique(wordDict[wordList[-1]])
        for entry in uniqueDict:
            probabilities.append(m[np.where(uniqueText == wordList[-1])[0][0]][np.where(uniqueText == entry)[0][0]])
        wordList.append(np.random.choice(uniqueDict, p=probabilities))

    print(' '.join(wordList) + '.')

generate_consitution_words(500)
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('words')
from nltk.corpus import words

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
pairsWords = make_pairs(listText)
wordTransition = transition_matrix(listText)
uniqueText = np.unique(listText)
wordDict = {}
for word_1, word_2 in pairsWords:
    if word_1 in wordDict.keys():
        wordDict[word_1].append(word_2)
    else:
        wordDict[word_1] = [word_2]

letterText = []
for letter in strText:
    if letter != '\n':
        letterText.append(letter)
letterTransition = transition_matrix(letterText)
uniqueLetters = np.unique(letterText)
pairsLetters = make_pairs(letterText)
letterDict = {}
for letter_1, letter_2 in pairsLetters:
    if letter_1 in letterDict.keys():
        letterDict[letter_1].append(letter_2)
    else:
        letterDict[letter_1] = [letter_2]

def generate_consitution_words(n):
    wordList = []
    first_word = "word"
    while first_word.islower() and first_word not in ['!', ',', '.', '?', ':', ';', '-', '(', ')', ' ']:
        first_word = np.random.choice(listText)
    wordList.append(first_word)

    for i in range(n):
        probabilities = []
        uniqueDict = np.unique(wordDict[wordList[-1]])
        for entry in uniqueDict:
            probabilities.append(wordTransition[np.where(uniqueText == wordList[-1])[0][0]][np.where(uniqueText == entry)[0][0]])
        wordList.append(np.random.choice(uniqueDict, p=probabilities))

    print(' '.join(wordList) + '.')
    return ' '.join(wordList) + '.'

def generate_constitution_letters(n):
    letterList = []
    first_letter = "w"
    while first_letter.islower() and first_letter not in ['!', ',', '.', '?', ':', ';', '-', '(', ')', ' ']:
            first_letter = np.random.choice(letterText)
    letterList.append(first_letter)

    count = n
    while count != 0:
        probabilities = []
        uniqueLetterDict = np.unique(letterDict[letterList[-1]])
        for entry in uniqueLetterDict:
            probabilities.append(letterTransition[np.where(uniqueLetters == letterList[-1])[0][0]][np.where(uniqueLetters == entry)[0][0]])
        letterList.append(np.random.choice(uniqueLetterDict, p=probabilities))
        if letterList[-1] == ' ':
            count = count - 1

    print(''.join(letterList))
    return ''.join(letterList)

letterstr = generate_constitution_letters(100)
setofwords = set(words.words())
wordCount = 0
for word in letterstr.split():    
    if word in setofwords:
        if len(word) != 1:
            wordCount = wordCount + 1
print("There are {} words here.".format(wordCount))
print('\n')
generate_consitution_words(500)
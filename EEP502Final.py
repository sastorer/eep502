import numpy as np
import pandas as pd
import re
import nltk
nltk.download('words')
from nltk.corpus import words

def transition_matrix(transitions):
    df = pd.DataFrame(transitions)
    # Create a new column with data shifted one space
    df['shift'] = df[0].shift(-1)
    # Add a count column (for group by function)
    df['count'] = 1
    # Groupby and then unstack, fill the zeros
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    # Normalise by occurences and save values to get transition matrix
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    return trans_mat

def transition_matrix_3d(transitions):
    df = pd.DataFrame(transitions)
    # Create a new column with data shifted one space
    df['shift'] = df[0].shift(-1)
    # Add a count column (for group by function)
    df['count'] = 1
    # Groupby and then unstack, fill the zeros
    trans_mat2d = df.groupby([0, 'shift']).count().unstack().fillna(0)
    # Normalise by occurences and save values to get transition matrix
    trans_mat2d = trans_mat2d.div(trans_mat2d.sum(axis=1), axis=0).values
    # Create a new column with data shifted two spaces
    df['shift2'] = df[0].shift(-2)
    # Groupby and then unstack, fill the zeros
    trans_mat3d = df.groupby([0, 'shift', 'shift2']).count().unstack().fillna(0)
    # Normalise by occurences and save values to get transition matrix
    trans_mat3d = trans_mat3d.div(trans_mat3d.sum(axis=1), axis=0)
    return trans_mat3d

def make_pairs(data):
    for i in range(len(data)-1):
        yield (data[i], data[i+1])

# Open the file and read in the text
strPath = "constitution.txt"
f = open(strPath)
strText = f.read()
# Put a space before any punctuation
strText = re.sub('([.,!?():;])', r' \1 ', strText)
# Remove double spaces
strText = re.sub('\s{2,}', ' ', strText)

# Split the long string into a list of words
listText = strText.split()
# Create the transition probability matrix
wordTransition = transition_matrix(listText)
# Create a list of unique words 
# (conveniently the same order that the transition matrix is in)
uniqueText = np.unique(listText)
# Create word pairs to enable key/value sorting in the next step
pairsWords = make_pairs(listText)
# Take the word pairs and sort them into a dictionary where the key 
# is one word and the values are all possible words that can follow
wordDict = {}
for word_1, word_2 in pairsWords:
    if word_1 in wordDict.keys():
        wordDict[word_1].append(word_2)
    else:
        wordDict[word_1] = [word_2]

# Take the original long string and separate it into a list of letters
letterText = []
for letter in strText:
    if letter != '\n':
        letterText.append(letter)
# create the transition probability matrix
letterTransition = transition_matrix(letterText)
# create a list of unique words
# (again conveniently the dame order that the transition matrix is in)
uniqueLetters = np.unique(letterText)
# Create letter pairs to enable key/value sorting in the next step
pairsLetters = make_pairs(letterText)
# Take the letter pairs and sort them into a dictionary where the key
# is one letter and the values are all possible letters that can follow
letterDict = {}
for letter_1, letter_2 in pairsLetters:
    if letter_1 in letterDict.keys():
        letterDict[letter_1].append(letter_2)
    else:
        letterDict[letter_1] = [letter_2]

letterPairs = []
actualPairs = []
for x in uniqueLetters:
    for y in uniqueLetters:
        letterPairs.append(x+y)
for pair in letterPairs:
    if strText.count(pair) > 0:
        actualPairs.append(pair)

def get_next_letter(word, sentence):
    index = 0
    dataAfter = []
    while index < len(sentence):
        index = sentence.find(word, index)
        if index == -1:
            break
        if index < len(sentence)-2:
            dataAfter.append(sentence[index+2])
        index += len(word)
    return dataAfter

pairsDict = {}
for pair in actualPairs:
    if pair in pairsDict.keys():
        for letter in get_next_letter(pair, ''.join(letterText)):
            pairsDict[pair].append(letter)
    else:
        pairsDict[pair] = []
        for letter in get_next_letter(pair, ''.join(letterText)):
            pairsDict[pair].append(letter)

doubleLetterTransition = transition_matrix_3d(letterText)

def generate_consitution_words(n):
    wordList = []
    first_word = "word"
    #Ensure the first word isn't a symbol and starts with a capital letter
    while first_word.islower() or first_word in ['!', ',', '.', '?', ':', ';', '-', '(', ')', ' ']:
        first_word = np.random.choice(listText)
    wordList.append(first_word)

    for i in range(n):
        # Build a list of probabilities to feed to np.random.choice 
        # (it needs to be the same length as the words to choose from)
        probabilities = []
        # Get the unique words that can follow the last word in word list 
        # (want to make sure all of the probabilities add up to 1)
        uniqueDict = np.unique(wordDict[wordList[-1]])
        # For each entry in that unique list of words, append the corresponding probability 
        # from the transition matrix to the list of probabilities
        for entry in uniqueDict:
            probabilities.append(wordTransition[np.where(uniqueText == wordList[-1])[0][0]][np.where(uniqueText == entry)[0][0]])
        # Finally, choose a word given all of the parameters!
        wordList.append(np.random.choice(uniqueDict, p=probabilities))

    # Print and return the final list of generated words in string form with a space between them
    print(' '.join(wordList) + '.')
    return ' '.join(wordList) + '.'

def generate_constitution_letters(n):
    letterList = []
    first_letter = "w"
    #Ensure the first letter isn't a symbol and starts with a capital letter
    while first_letter.islower() or first_letter in ['!', ',', '.', '?', ':', ';', '-', '(', ')', ' ']:
            first_letter = np.random.choice(letterText)
    letterList.append(first_letter)

    count = n
    while count != 0:
        # Build a list of probabilities to feed to np.random.choice 
        # (it needs to be the same length as the letters to choose from)
        probabilities = []
        # Get the unique letters that can follow the last letter in letter list 
        # (want to make sure all of the probabilities add up to 1)
        uniqueLetterDict = np.unique(letterDict[letterList[-1]])
        # For each entry in that unique list of letters, append the corresponding probability 
        # from the transition matrix to the list of probabilities
        for entry in uniqueLetterDict:
            probabilities.append(letterTransition[np.where(uniqueLetters == letterList[-1])[0][0]][np.where(uniqueLetters == entry)[0][0]])
        # Finally, choose a word given all of the parameters!
        letterList.append(np.random.choice(uniqueLetterDict, p=probabilities))
        # If that choice is a space, consider the preceeding choices a word and deduct from the word count
        if letterList[-1] == ' ':
            count = count - 1

    # Print and return the final list of generated letters in string form with nothing between them
    print(''.join(letterList))
    return ''.join(letterList)

def generate_constitution_letters_double(n):
    # Start with a pair of letters that we know is in the corpus
    letterList = ['W','e']
    count = n
    while count != 0:
        # Build a list of probabilities to feed to np.random.choice 
        # (it needs to be the same length as the letters to choose from)
        probabilities = []
        # Get the unique letters that can follow the last pair of letters in letter list 
        # (want to make sure all of the probabilities add up to 1)
        pair = letterList[-2]+letterList[-1]
        if pair in pairsDict.keys():
            uniqueLetterDict = np.unique(pairsDict[pair])
            # For each entry in that unique list of letters, append the corresponding probability 
            # from the transition matrix to the list of probabilities
            for entry in uniqueLetterDict:
                if np.where(uniqueLetters == entry)[0].size != 0:
                    probabilities.append(doubleLetterTransition.loc[(letterList[-2], letterList[-1])][(np.where(uniqueLetters == entry)[0][0])])
            # Finally, choose a word given all of the parameters!
            letterList.append(np.random.choice(uniqueLetterDict, p=probabilities))
            # If that choice is a space, consider the preceeding choices a word and deduct from the word count
            if letterList[-1] == ' ':
                count = count - 1
        else:
            # If we made a bad choice and that pair isn't in our dictionary, remove the last guess and try again
            letterList = letterList[:-1]

    # Print and return the final list of generated letters in string form with nothing between them
    print(''.join(letterList))
    return ''.join(letterList)

letterstr = generate_constitution_letters(100)

# Check how many words are in the generated string of letters
setofwords = set(words.words())
wordCount = 0
for word in letterstr.split():    
    if word in setofwords:
        if len(word) != 1:
            wordCount = wordCount + 1
print("There are {} words here.".format(wordCount))
print('\n')

doubleletterstr = generate_constitution_letters_double(100)

# Check how many words are in the generated string of letters
wordCountDouble = 0
for word in doubleletterstr.split():    
    if word in setofwords:
        if len(word) != 1:
            wordCountDouble = wordCountDouble + 1
print("There are {} words here.".format(wordCountDouble))
print('\n')
generate_consitution_words(500)
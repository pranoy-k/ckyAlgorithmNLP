

import sys
import numpy as np
from pprint import pprint

"""
#The simple logic to write the program is pi[i, j,X] = maximum probability
of a constituent with non-terminal X spanning words i . . . j inclusive
## Our goal is to find pi[1,n,S] 
### Base case is to define all words if they are present in the lexicons
#### The import one, the recursive definition pi(i,j,X) = 
max(q(x->yz)*pi(i,t,y)*pi(t+1,j,z))
given x->yz is in the grammar and t belongs to i:j-1
"""
class Unary_Rule:
    def __init__(self,*rule):
        self.left, self.right = rule[0],rule[1]
        self.prob = rule[2]

class Binary_Rule:
    def __init__(self,*rule):
        self.left, self.right1, self.right2 = rule[0],rule[1],rule[2]
        self.prob = rule[3]
class Lexicon:
    def __init__(self,*rule):
        self.left, self.word = rule[0],rule[1]
        self.prob = rule[2]

def preProcessGrammar(grammar_rules):
    """
    Preprocessing grammar_rules into unary and binaries.
    """

    Unaries = []
    Binaries = []
    Lexicons = []
    for i, rule in grammar_rules.items():
        temp = rule.split()
        # print("debug: Pranoy")
        if(len(temp) == 3):
            if(temp[1][0] >= 'a' and temp[1][0] <= 'z'):
                lexicon = Lexicon(*temp)
                Lexicons.append(lexicon)
            else:
                # print("debug: ",temp[2])
                unary = Unary_Rule(*temp)
                Unaries.append(unary)
        else:
            binary = Binary_Rule(*temp)
            Binaries.append(binary)

    return Binaries, Unaries, Lexicons

def getNonTerminals(Binaries,Unaries,Lexicons):
    N = set()
    for rule in Unaries:
        N.add(rule.left)
        N.add(rule.right)
    for rule in Binaries:
        N.add(rule.left)
        N.add(rule.right1)
        N.add(rule.right2)
    for rule in Lexicons:
        N.add(rule.left)

    return dict(zip(list(N),list(range(len(N)))))

def searchInLexicons(n, word, Lexicons):
    for Lexicon in Lexicons:
        if(Lexicon.left == n and Lexicon.word == word):
            return True, float(Lexicon.prob)
    return False,0

def searchInUnaries(n1, n2, Unaries):
    for unary in Unaries:
        if(unary.left == n1 and unary.right == n2):
            return True, float(unary.prob)
    return False,0  

def searchInBinaries(n1, n2, n3, Binaries):
    for Lexicon in Lexicons:
        if(Lexicon.left == n and Lexicon.word == word):
            return True, Lexicon.prob
    return False,0

def shuffleNonTerminals(N, number):
    """
    Create all possible combinations of the non-teminals in 
    number X number X number ways
    """

    length = len(N.keys())

    S = []  ## S is for the shuffled list
    if(number == 2):
        for n1 in N.keys():
            for n2 in N.keys():
                S.append([n1,n2])

    if(number == 3):
        for n1 in N.keys():
            for n2 in N.keys():
                for n3 in N.keys():
                    S.append([n1,n2,n3])

    return S



def ckyAlgorithm(sentence, grammar_rules):
    """
    inputs to our algorithm
    sentence: x1,x2, ....  xn
    PCFG: G = [N,Sigma,S,R,q]
    N: set of non-terminals
    Sigma: set of terminals
    S: start symbol of the grammar
    R: Set of Rules

    Algorithm: 

    Loop: iterate over the span: span varies from 1 to n-1
        Loop: iterate over i, i varies from 1 to n-l,  here l is the span variable
            set j = i + l, since j is the span limit
                Loop: for all non-terminal N setting them as X
                    pi(i,j,X) = max(q(X->YZ)*pi(i,t,Y)*pi(t+1,j,Z)) ,, t belongs to i:j-1
                    and Q(X->YZ) is in the grammar

    returns [mostProbableParse, probability]
    """
    Binaries, Unaries, Lexicons = preProcessGrammar(grammar_rules)
    # print("\ndebug: Printing all Grammar Rules \n", preProcessGrammar(grammar_rules),"\n")

    N = getNonTerminals(Binaries,Unaries,Lexicons)
    num_words = len(sentence.split())
    words = sentence.split()
    score = [[[0 for i in range(len(N))] for j in range(num_words + 1)] for k in range(num_words + 1)]
    back = [[[[] for i in range(len(N))] for j in range(num_words + 1)] for k in range(num_words + 1)]

    # print("debug: score.shape", np.array(score).shape)

    ## For Unaries 
    # print("The non-terminals are :",N)
    for i,word in enumerate(words):
        print("SPAN:", word)
        for n in N.keys():
            found, prob =  searchInLexicons(n, word, Lexicons)
            if found and i < num_words - 1:
                score[i][i+1][N[n]] = (prob)

        
        

        added = True
        while added:
            added = False
            # i=0
            for temp in shuffleNonTerminals(N,2):
                A, B  = temp[0],temp[1]
                # print("debug ",A,B, i ) ; i+=1
                # print(":debug ",type(score[i][i+1][N[B]]))
                if i < num_words - 1:
                    if (score[i][i+1][N[B]] > 0) and (searchInUnaries(A, B, Unaries)[0]):
                        # print("debug: It comes inside")
                        prob = searchInUnaries(A, B, Unaries)[1] * score[i][i+1][N[B]]
                        if prob > score[i][i+1][N[A]]:
                            # print("debug: It comes inside")
                            score[i][i+1][N[A]] = (prob)
                            back[i][i+1][N[A]] = B
                            added = True
        for ii in range(len(N)):
            if(score[i][i+1][ii] != 0):
                print("P(",list(N.keys())[list(N.values()).index(ii)],") =",round(score[i][i+1][ii],2),"(BackPointer =",back[i][i+1][ii])

                #P(NP) = 0.14 (BackPointer = N)
            # assert False
        # print("\n\ndebug: The score after editing diagonal elements", score)



    





def main():
    grammar_rules = {}
    sents = []
    # print("debug: Grammar Rules :::: \n")

    with open(sys.argv[1]) as file:
        for i,line in enumerate(file):
            grammar_rules[i] = line
            # print("debug: Grammar Rules",i,":",line)
            
    # print("\debug: n\nSentences :::: \n")
    with open(sys.argv[2]) as file:
        for i,line in enumerate(file):
            sents.append(line)
            # print("debug: Sentence",i,":",line)


    for sent in sents:
        print("PROCESSING SENTENCE:",sent,'\n')
        ckyAlgorithm(sent, grammar_rules)
## Dynamic Programming Task 
# How will you do it!!!??

main()

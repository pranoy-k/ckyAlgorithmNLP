

import sys


#The simple logic to write the program is pi[i, j,X] = maximum probability of a constituent with non-terminal X spanning words i . . . j inclusive
## Our goal is to find pi[1,n,S] 
### Base case is to define all words if they are present in the lexicons
#### The import one, the recursive definition pi(i,j,X) = max(q(x->yz)*pi(i,t,y)*pi(t+1,j,z)) given x->yz is in the grammar and t belongs to i:j-1


def ckyAlgorithm(sentence, grammar_rules):
	


def main():
	grammar_rules = {}
	sents = []
	print("Debug ", sys.argv)

	with open(sys.argv[1]) as file:
		for i,line in enumerate(file):
			grammar_rules[i] = line
			

	with open(sys.argv[2]) as file:
		for line in file:
			sents.append(line)


	for sent in sents:
		print("PROCESSING SENTENCE:",sent,'\n')
		ckyAlgorithm(sentence, grammar_rules)
## Dynamic Programming Task 
# How will you do it!!!??

main()

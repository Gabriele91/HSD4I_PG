#these lines are required to not print a "benign" warning
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import featuresGenerator
import numpy as np
import sklearn
import sys


#filter out features
def filter(postsFilename,kbest=500):
	posts = featuresGenerator.loadFeatures(postsFilename)
	X,y = posts2npArrays(posts)
	if X.shape[1]<=kbest: return posts
	selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.mutual_info_classif,k=kbest)
	selector.fit(X,y)
	Xnew = selector.transform(X)
	npArrays2posts(posts,Xnew)
	return posts


#get X,y from posts
def posts2npArrays(posts):
	X = np.stack([ p['features'] for p in posts ])
	y = np.stack([ p['label'] for p in posts ])
	return X,y


#get posts from X,y
def npArrays2posts(posts,X):
	for i in range(X.shape[0]):
		posts[i]['features'] = X[i]


#print usage
def usage():
	print('USAGE: python3 featuresFilter.py INPUT_FILE OUTPUT_FILE [-kbest <NUM>]',file=sys.stderr)
	print('DEFAULT:',file=sys.stderr)
	print('\t-kbest  500',file=sys.stderr)


#main part
if __name__ == '__main__':
	if len(sys.argv)<3:
		usage()
		sys.exit()
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
	kbest = 500
	verbose = True
	i = 3
	while i<len(sys.argv):
		if sys.argv[i] in {'-kbest', '-best', '-k'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -kbest requires <NUM> argument")
			kbest = int(sys.argv[i+1])
		elif sys.argv[i] in {'-verbose', '-v'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -verbose requires y|n argument")
			verbose = sys.argv[i+1][0] not in 'nNfF0'
		i += 2
	if verbose: print('Selecting the best '+str(kbest)+' features from '+inputFile+' ...')
	posts = filter(inputFile,kbest)
	featuresGenerator.saveFeatures(posts,outputFile)
	if verbose: print('Done. Reduced features have been saved in '+outputFile+'.')

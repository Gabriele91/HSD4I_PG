#these lines are required to not print a "benign" warning
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import featuresGenerator
import featuresFilter
import sklearn.svm
import sklearn.model_selection
import sklearn.metrics
import sys
import pickle
import sklearn.pipeline

def loadFeatures(featuresFilename):
	if verbose: print('* Loading features from '+featuresFilename+' ...')
	posts = featuresGenerator.loadFeatures(featuresFilename)
	X,y = featuresFilter.posts2npArrays(posts)
	if verbose: print('  Features from '+featuresFilename+' have been loaded.')
	return X,y,posts

def trainClassifier(X, y , seed=1,verbose=True,C=2.0,gamma='auto',preprocess=None,kernel='rbf'):
	if preprocess is None:		scaler = None
	elif preprocess=='norm':	scaler = sklearn.preprocessing.Normalizer()
	elif preprocess=='stand':	scaler = sklearn.preprocessing.StandardScaler()
	elif preprocess=='rstand':	scaler = sklearn.preprocessing.RobustScaler()
	if scaler is not None:
		scaler=scaler.fit(X)
		X=scaler.transform(X)	
	classifier = sklearn.svm.SVC(kernel=kernel,C=C,gamma=gamma,random_state=seed,verbose=(20 if verbose else 0),class_weight='balanced')
	classifier=classifier.fit(X,y)
	return scaler, classifier

def classify(scaler, classifier, X):
	if scaler is not None:
		X=scaler.transform(X)
	y=classifier.predict(X)
	return y

#launch the cross validation experiment and return metrics (f1s,acc,pre,rec,cm)
def run(trainFeaturesFilename, testFeaturesFilename, outputFileName,seed=1,verbose=True,C=2.0,gamma='auto',preprocess=None,kernel='rbf'):
	X, y, posts = loadFeatures(trainFeaturesFilename)
	X_Test, y_dummy, testPosts = loadFeatures(testFeaturesFilename)
	scaler, classifier = trainClassifier (X, y, seed=seed, verbose = verbose,C=C,gamma=gamma,preprocess=preprocess,kernel=kernel)
	y_pred = classify(scaler,classifier,X_Test)
	with open(outputFileName,'w') as f:
		for i,post in enumerate(testPosts):
			print(testPosts[i]['id']+"\t"+testPosts[i]['text']+"\t"+str(y_pred[i]),file=f)

#print usage
def usage():
	print('USAGE: python3 classifier_testset.py TRAINING_FILE TEST_FILE OUTPUT_FILE [-verbose y|n] [-C <FLOAT>] [-gamma auto|<FLOAT>] [-nfolds <NUM>] [-seed <NUM>] [-kernel rbf|linear] [-preprocess norm|stand|rstand]',file=sys.stderr)
	print('DEFAULT:',file=sys.stderr)
	print('\t-verbose    y',file=sys.stderr)
	print('\t-C          2.0',file=sys.stderr)
	print('\t-gamma      auto',file=sys.stderr)
	print('\t-nfolds     10',file=sys.stderr)
	print('\t-seed       1',file=sys.stderr)
	print('\t-preprocess none',file=sys.stderr)
	print('\t-kernel     rbf',file=sys.stderr)


#main part
if __name__ == '__main__':
	if len(sys.argv)<3:
		usage()
		sys.exit()
	trainingFileName = sys.argv[1]
	testFileName = sys.argv[2]
	outputFileName = sys.argv[3]
	verbose = True
	C = 2.0
	gamma = 'auto'
	seed = 1
	preprocess = None
	kernel = 'rbf'
	i = 4
	while i<len(sys.argv):
		if sys.argv[i] in {'-verbose', '-v'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -verbose requires y|n argument")
			verbose = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-C', '-c'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -C requires <FLOAT> argument")
			C = float(sys.argv[i+1])
		elif sys.argv[i] in {'-gamma', '-g'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -gamma requires auto|<FLOAT> argument")
			gamma = 'auto' if sys.argv[i+1][0]=='a' else float(sys.argv[i+1])
		elif sys.argv[i] in {'-seed', '-s'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -seed requires <NUM> argument")
			seed = int(sys.argv[i+1])
		elif sys.argv[i] in {'-preprocess', '-pp'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -preprocess requires none|norm|stand|rstand argument")
			preprocess = None if sys.argv[i+1]=='none' else sys.argv[i+1]
		elif sys.argv[i]=='-kernel':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -kernel requires rbf|linear argument")
			kernel = sys.argv[i+1]
		i += 2
	if verbose: print('Running experiment ...')
	run(trainingFileName,testFileName,outputFileName,seed=seed,verbose=verbose,C=C,gamma=gamma,preprocess=preprocess,kernel=kernel)
	print("DONE")

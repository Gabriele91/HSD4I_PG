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


#classifier instance
classifier = None


#launch the cross validation experiment and return metrics (f1s,acc,pre,rec,cm)
def run(featuresFilename,seed=1,verbose=True,nfolds=10,C=2.0,gamma='auto',preprocess=None,kernel='rbf'):
	global classifier
	if verbose: print('* Loading features from '+featuresFilename+' ...')
	posts = featuresGenerator.loadFeatures(featuresFilename)
	X,y = featuresFilter.posts2npArrays(posts)
	if verbose: print('  Features from '+featuresFilename+' have been loaded.')
	if verbose: print('* Running cross validation using stratified k-fold with k='+str(nfolds)+' ...')
	classifier = sklearn.svm.SVC(kernel=kernel,C=C,gamma=gamma,random_state=seed,verbose=(20 if verbose else 0))
	if preprocess is None:
		estimatorsList = [ ('classifier',classifier) ]
	else:
		scaler = None
		if preprocess=='norm':		scaler = sklearn.preprocessing.Normalizer()
		elif preprocess=='stand':	scaler = sklearn.preprocessing.StandardScaler()
		elif preprocess=='rstand':	scaler = sklearn.preprocessing.RobustScaler()
		estimatorsList = [ ('scaler',scaler), ('classifier',classifier) ]
	pipeline = sklearn.pipeline.Pipeline(estimatorsList)
	cv = sklearn.model_selection.StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=seed)
	#y_pred = sklearn.model_selection.cross_val_predict(classifier,X,y,cv=cv,n_jobs=-1,verbose=(20 if verbose else 0))
	y_pred = sklearn.model_selection.cross_val_predict(pipeline,X,y,cv=cv,n_jobs=-1,verbose=(20 if verbose else 0))
	if verbose: print('  Cross validation done.')
	if verbose: print('* Computing metrics ...')
	cm = sklearn.metrics.confusion_matrix(y,y_pred)
	t0,f0,t1,f1 = cm[0,0],cm[1,0],cm[1,1],cm[0,1]
	#p1,r1 = t1/(t1+f1), t1/(t1+f0)
	#p0,r0 = t0/(t0+f0), t0/(t0+f1)
	#f1s1 = 2*((p1*r1)/(p1+r1))
	#f1s0 = 2*((p0*r0)/(p0+r0))
	#f1s_ma = (f1s1+f1s0)/2
	f1s = sklearn.metrics.f1_score(y,y_pred,average='macro')
	pre = sklearn.metrics.precision_score(y,y_pred,average='macro')
	rec = sklearn.metrics.recall_score(y,y_pred,average='macro')
	acc = sklearn.metrics.accuracy_score(y,y_pred)
	if verbose:
		print('  Metrics computed:')
		print('    - f1-score:    '+str(f1s))
		print('    - accuracy:    '+str(acc))
		print('    - precision:   '+str(pre))
		print('    - recall:      '+str(rec))
		#print('    - t0,f0,t1,f1: '+str(t0)+','+str(f0)+','+str(t1)+','+str(f1))
		print('    - confusion matrix:')
		print(cm)
		print('    - classification report from sklearn:')
		print(sklearn.metrics.classification_report(y,y_pred))
	return {'f1score':f1s, 'accuracy':acc, 'precision':pre, 'recall':rec, 'confusion_matrix':cm}


#save the metrics using pickle
def saveMetrics(filename,metrics):
	fout = open(filename,'wb')
	pickle.dump(metrics,fout)
	fout.close()


#load matrics using pickle
def loadMetrics(filename):
	fin = open(filename,'rb')
	metrics = pickle.load(fin)
	fin.close()
	return metrics


#train the classifier with X features and y classes
#def train(X,y,C=1.0,gamma='auto'): #gamma='auto' means 1/n_features
#	global classifier
#	classifier = sklearn.svm.SVC(kernel='rbf',C=1.0,gamma='auto')
#	classifier = classifier.fit(X,y)


#predict a single array of features
#def predict(z):
#	return classifier.predict([z])


#predict a list of arrays of features
#def predictAll(zlist):
#	return classifier.predict(zlist)


#print usage
def usage():
	print('USAGE: python3 classifier.py INPUT_FILE OUTPUT_FILE [-verbose y|n] [-C <FLOAT>] [-gamma auto|<FLOAT>] [-nfolds <NUM>] [-seed <NUM>] [-kernel rbf|linear] [-preprocess norm|stand|rstand]',file=sys.stderr)
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
	inputFilename = sys.argv[1]
	outputFilename = sys.argv[2]
	verbose = True
	C = 2.0
	gamma = 'auto'
	nfolds = 10
	seed = 1
	preprocess = None
	kernel = 'rbf'
	i = 3
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
		elif sys.argv[i] in {'-nfolds', '-n', '-k', '-kfolds'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -nfolds requires <NUM> argument")
			nfolds = int(sys.argv[i+1])
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
	metrics = run(inputFilename,seed=seed,verbose=verbose,nfolds=nfolds,C=C,gamma=gamma,preprocess=preprocess,kernel=kernel)
	if verbose: print(' ... experiment finished.')
	if verbose: print('Saving metrics in '+outputFilename+' ...')
	saveMetrics(outputFilename,metrics)
	if verbose: print('Metrics saved in '+outputFilename+'.')

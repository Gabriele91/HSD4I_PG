#these lines are required to not print a "benign" warning
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import util
import tokenizer
import gensim.models
import gensim.models.callbacks
import sys
import time


#buildup a word-vector model using FastText
def buildModel(
		postsFile,tokenized=True,stemming=True,stopwords=True,
		modelToUpdate=None,sg=1,size=300,min_n=3,max_n=6,window=5,iter=50,alpha=0.025,min_alpha=0.0001,hs=0,negative=5,seed=1,workers=3,
		verbose=True
	):
	if verbose: print('Loading '+postsFile+' ...')
	stdin = postsFile in {'stdin','-'}
	fposts = sys.stdin if stdin else open(postsFile,'r',encoding=util.getFileEncoding(postsFile))
	posts = [ line.strip().split(' ') if tokenized else tokenizer.tokenize(line.strip(),stemming,stopwords) for line in fposts ]
	if not stdin: fposts.close()
	if verbose: print(' ... '+postsFile+' has been loaded. FastText training is starting ...')
	if modelToUpdate is not None and verbose: print('Loading the model to update from '+modelToUpdate+' ...')
	model = loadModel(modelToUpdate) if modelToUpdate is not None else gensim.models.FastText(min_count=1,size=size,min_n=min_n,max_n=max_n,window=window,iter=iter,alpha=alpha,min_alpha=min_alpha,hs=hs,negative=negative,seed=seed,workers=workers)
	if modelToUpdate is not None and verbose: print(' ... model to update loaded.')
	if verbose: model.callbacks = [EpochLogger(iter)]
	model.build_vocab(posts,update=modelToUpdate is not None)
	model.train(posts,total_examples=model.corpus_count,epochs=model.iter,compute_loss=True)
	if verbose:
		print(' ... FastText finished!')
		model.callbacks = ()
	return model


#save a word-vector model to a file
def saveModel(model,filename):
	model.save(filename)


#load a word-vector model from a file
def loadModel(filename):
	if filename[-4:]=='.bin': #model saved from Facebook original FastText implementation (IT IS NOT POSSIBLE TO CONTINUE TRAINING THIS MODEL!!!)
		return gensim.models.FastText.load_fasttext_format(filename[:-4])
	else: #model saved using gensim
		return gensim.models.FastText.load(filename)


#callbacks to track the training
class EpochLogger(gensim.models.callbacks.CallbackAny2Vec):

	def __init__(self,nepochs):
		self.epoch = 1
		self.nepochs = nepochs
	
	def on_epoch_begin(self, model):
		print('{}/{} epochs done. Loss={}. Working for epoch {} ...'.format(self.epoch-1,self.nepochs,model.running_training_loss,self.epoch),end='')

	def on_epoch_end(self, model):
		print('{}/{} epochs done. Loss={}.'.format(self.epoch,self.nepochs,model.running_training_loss),end=' '*45)
		self.epoch += 1


#print instructions for command line
def usage():
	print('USAGE: python3 wvModelBuilder.py INPUT_FILE OUTPUT_FILE [-verbose y|n] [-update <FILE_NAME>] [-from_line <NUM>] [-to_line <NUM>|LAST] [-tokenized y|n] [-stemming y|n] [-stopwords y|n] [-sg 0|1] [-size <NUM>] [-min_n <NUM>] [-max_n <NUM] [-window <NUM>] [-iter <NUM>] [-alpha <FLOAT>] [-min_alpha <FLOAT>] [-hs 0|1] [-negative <NUM>] [-seed <NUM>] [-workers <NUM>]',file=sys.stderr)
	print('DEFAULT:',file=sys.stderr)
	print('\t-verbose   y',file=sys.stderr)
	print('\t-update    None',file=sys.stderr)
	print('\t-tokenized y',file=sys.stderr)
	print('\t-stemming  y',file=sys.stderr)
	print('\t-stopwords y',file=sys.stderr)
	print('\t-sg        1',file=sys.stderr)
	print('\t-size      300',file=sys.stderr)
	print('\t-min_n     3',file=sys.stderr)
	print('\t-max_n     6',file=sys.stderr)
	print('\t-window    5',file=sys.stderr)
	print('\t-iter      50',file=sys.stderr)
	print('\t-alpha     0.025',file=sys.stderr)
	print('\t-min_alpha 0.0001',file=sys.stderr)
	print('\t-hs        0',file=sys.stderr)
	print('\t-negative  5',file=sys.stderr)
	print('\t-seed      1',file=sys.stderr)
	print('\t-workers   3',file=sys.stderr)
	print('NOTE_1: it can read utf-8 and latin_1 (ISO-8859) encodings.',file=sys.stderr)
	print('NOTE_2: INPUT_FILE can be set to "stdin" (or "-").',file=sys.stderr)


#main part of the module
if __name__ == '__main__':
	if len(sys.argv)<3:
		usage()
		sys.exit()
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
	verbose = True
	modelToUpdate = None
	tokenized = True
	stemming = True
	stopwords = True
	sg = 1
	size = 300
	min_n = 3
	max_n = 6
	window = 5
	iter = 50
	alpha = 0.025
	min_alpha = 0.0001
	hs = 0
	negative = 5
	seed = 1
	workers = 3
	i = 3
	while i<len(sys.argv):
		if sys.argv[i] in {'-verbose', '-v'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -verbose requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': verbose = False
		elif sys.argv[i] in {'-update', '-upd', '-model'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -update requires <FILE_NAME> argument")
			modelToUpdate = sys.argv[i+1]
		elif sys.argv[i] in {'-tokenized', '-tok'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -tokenized requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': tokenized = False
		elif sys.argv[i] in {'-stemming', '-stem'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stemming requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': stemming = False
		elif sys.argv[i] in {'-stopwords', '-sw'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stopwords requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': stopwords = False
		elif sys.argv[i]=='-sg':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -sg requires 1|0 argument")
			sg = int(sys.argv[i+1])
		elif sys.argv[i]=='-size':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -size requires <NUM> argument")
			size = int(sys.argv[i+1])
		elif sys.argv[i] in {'-min_n', '-minn'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -min_n requires <NUM> argument")
			min_n = int(sys.argv[i+1])
		elif sys.argv[i] in {'-max_n', '-maxn'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -max_n requires <NUM> argument")
			max_n = int(sys.argv[i+1])
		elif sys.argv[i] in {'-window', '-win'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -window requires <NUM> argument")
			window = int(sys.argv[i+1])
		elif sys.argv[i] in {'-iter', '-epochs'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -iter requires <NUM> argument")
			iter = int(sys.argv[i+1])
		elif sys.argv[i] in {'-alpha', '-lr'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -alpha requires <FLOAT> argument")
			alpha = float(sys.argv[i+1])
		elif sys.argv[i] in {'-min_alpha', '-minalpha', '-min_lr', '-minlr'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -min_alpha requires <FLOAT> argument")
			min_alpha = float(sys.argv[i+1])
		elif sys.argv[i]=='-hs':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -hs requires 1|0 argument")
			hs = int(sys.argv[i+1])
		elif sys.argv[i] in {'-negative', '-neg'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -negative requires 1|0 argument")
			hs = int(sys.argv[i+1])
		elif sys.argv[i]=='-seed':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -seed requires <NUM> argument")
			seed = int(sys.argv[i+1])
		elif sys.argv[i]=='-workers':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -workers requires <NUM> argument")
			workers = int(sys.argv[i+1])		
		i += 2
	if verbose: print('Building word-vector model from '+inputFile+' ...')
	t0 = time.process_time()
	model = buildModel(
				inputFile,tokenized=tokenized,stemming=stemming,stopwords=stopwords,
				modelToUpdate=modelToUpdate,sg=sg,size=size,min_n=min_n,max_n=max_n,window=window,iter=iter,alpha=alpha,min_alpha=min_alpha,hs=hs,negative=negative,seed=seed,workers=workers,
				verbose=verbose
			)
	saveModel(model,outputFile)
	if verbose: print('\nDone in '+str(round(time.process_time()-t0,3))+' seconds. Model has been saved in '+outputFile)

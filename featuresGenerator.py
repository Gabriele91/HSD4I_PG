#these lines are required to not print a "benign" warning
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


import util
import tokenizer
import wvModelBuilder
import math
import numpy as np
import sklearn.preprocessing
import sys
import re
import aspell
import pickle
import featuresFilter


#some useful variables
re_ht = re.compile('#[\w\d]+')
re_words = re.compile('[\w\d]+')
re_dots = re.compile('\.\.+') #2 or more dots
speller = aspell.Speller('lang','it')
stemmedStopwords = None


#generate all the features
def generateFeatures(postsFilename,wvModelFilename,
						hatesvFilename=None,sentixFilename=None,
						stemming=True,stopwords=True,
						favg=True,fstd=False,fmin=False,fmax=False,fmed=False,fsum=False,
						fdavg=False,fdstd=False,fdmin=False,fdmax=False,fdmed=False,fdsum=False,
						idfWeights=False,
						extraFeatures=True,
						verbose=True):
	#loading original posts + tokenization + label in a list of dictionaries
	if verbose: print('* Reading and tokenizing '+postsFilename+' ...')
	fposts = open(postsFilename,'r',encoding=util.getFileEncoding(postsFilename))
	posts = [ { 'id': id, 'text': post, 'tokens': tokenizer.tokenize(post,stemming,stopwords), 'label': int(label) } for i,id,post,label in util.scanFileLines(fposts,tsv=True) ]
	fposts.close()
	if verbose: print('  Posts from '+postsFilename+' have been read and tokenized.')
	#loading word-vector model from FastText
	if verbose: print('* Loading word-vector model from '+wvModelFilename+' ...')
	wvModel = wvModelBuilder.loadModel(wvModelFilename)
	if verbose: print('  Word-vector model from '+wvModelFilename+' has been loaded.')
	#computing idf weights for every unique token
	idf = None
	if idfWeights:
		if verbose: print('* Computing idf weights ...')
		idf = computeIdfWeights(posts)
		if verbose: print('  Idf weights for '+str(len(idf))+' unique tokens have been computed.')
	#load hate speech vocabulary
	if extraFeatures:
		if verbose: print('* Loading hate speech vocabulary from '+hatesvFilename+' ...')
		fhatesv = open(hatesvFilename,'r',encoding=util.getFileEncoding(hatesvFilename))
		hatesv = [ tokenizer.stem(line.strip()) if stemming else line.strip() for line in fhatesv ]
		fhatesv.close()
		if verbose: print('  Hate speech vocabulary from '+hatesvFilename+' has been loaded.')
	#load sentix sentiment data
	if extraFeatures:
		if verbose: print('* Loading sentiment data from '+sentixFilename+' ...')
		fsentix = open(sentixFilename,'r',encoding=util.getFileEncoding(sentixFilename))
		sentix = {}
		for line in fsentix:
			l = line.strip().split('\t')
			lemma = l[0].lower()
			if stemming: lemma = tokenizer.stem(lemma)
			polar,inten = float(l[-2]),float(l[-1])
			if lemma not in sentix:
				sentix[lemma] = [polar,inten,1]
			else:
				nn = sentix[lemma][2]
				newpolar = (sentix[lemma][0]*nn + polar)/(nn+1)
				newinten = (sentix[lemma][1]*nn + inten)/(nn+1)
				sentix[lemma] = [newpolar,newinten,nn+1]
		fsentix.close()
		if verbose: print('  Sentiment data from '+sentixFilename+' have been loaded.')
	#create numeric features for all the posts
	if verbose: print('* Computing numeric features ...')
	for i in range(len(posts)):
		features1 = aggregateWordVectors(posts[i]['tokens'],wvModel,idf,favg,fstd,fmin,fmax,fmed,fsum,fdavg,fdstd,fdmin,fdmax,fdmed,fdsum)
		posts[i]['features'] = np.concatenate((features1,generateExtraFeatures(posts[i]['text'],posts[i]['tokens'],hatesv,sentix,stemming,stopwords,features1.dtype))) if extraFeatures else features1
	if verbose: print('  Numeric features have been computed.')
	#return the list of dictionaries containing (text+tokens+label+features)
	return posts


#save the posts with the features using pickle
def saveFeatures(posts,filename):
	fout = open(filename,'wb')
	pickle.dump(posts,fout)
	fout.close()


#load the posts with the features saved using pickle
def loadFeatures(filename):
	fin = open(filename,'rb')
	posts = pickle.load(fin)
	fin.close()
	return posts


#compute idf weights for every unique token
def computeIdfWeights(posts):
	#using the main idf formula in https://en.wikipedia.org/wiki/Tf%E2%80%93idf with denominator adjustment
	nposts = len(posts)
	idf = {}
	for p in posts:
		for t in set(p['tokens']):
			if t not in idf: idf[t] = 1
			else: idf[t] += 1
	for t in idf:
		idf[t] = math.log(nposts/idf[t])
	return idf


#aggregate the word vectors
def aggregateWordVectors(tokens,wvModel,idf,favg,fstd,fmin,fmax,fmed,fsum,fdavg,fdstd,fdmin,fdmax,fdmed,fdsum):
	#get word-vector size and dtype
	tmp = wvModel.wv['test']
	wv_size = tmp.shape[0]
	dtype = tmp.dtype
	tmp = None
	#compute numeric tokens as a numpy 2d array (1 row per token) by using wvModel
	m = np.stack( [wvModel.wv[token]*(idf[token] if idf is not None else 1.) for token in tokens] if len(tokens)>0 else [np.zeros(wv_size,dtype=dtype)] )
	#m = np.stack(  [np.array(np.random.random([wv_size]),dtype=dtype) for token in tokens ] if len(tokens)>0 else [np.zeros(wv_size,dtype=dtype)] )
	#initialize empty numpy arrays
	vavg,vstd,vmin,vmax,vmed,vsum = np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype)
	vdavg,vdstd,vdmin,vdmax,vdmed,vdsum = np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype),np.empty(0,dtype=dtype)
	#set the 'normal' features by aggregating the column values of m
	if favg: vavg = m.mean(0)
	if fstd: vstd = m.std(0)
	if fmin: vmin = m.min(0)
	if fmax: vmax = m.max(0)
	if fmed: vmed = np.median(m,0)
	if fsum: vsum = m.sum(0)
	#if one of the 'difference' features has to be computed
	if fdavg or fdstd or fdmin or fdmax or fdmed or fdsum:
		#compute the matrix of consecutive row differences
		md = np.diff(m,axis=0) if m.shape[0]>1 else None
		#set the 'difference' features by aggregating the column values of md
		if fdavg: vdavg = md.mean(0) if md is not None else np.zeros(wv_size,dtype=dtype)
		if fdstd: vdstd = md.std(0) if md is not None else np.zeros(wv_size,dtype=dtype)
		if fdmin: vdmin = md.min(0) if md is not None else np.zeros(wv_size,dtype=dtype)
		if fdmax: vdmax = md.max(0) if md is not None else np.zeros(wv_size,dtype=dtype)
		if fdmed: vdmed = np.median(md,0) if md is not None else np.zeros(wv_size,dtype=dtype)
		if fdsum: vdsum = md.sum(0) if md is not None else np.zeros(wv_size,dtype=dtype)
	#return the concatenation of the numpy arrays
	return np.concatenate((vavg,vstd,vmin,vmax,vmed,vsum,vdavg,vdstd,vdmin,vdmax,vdmed,vdsum))


#generate extra features for the post/tokens
def generateExtraFeatures(text,tokens,hatesv,sentix,stemming,stopwords,dtype=np.dtype('float32')):
	#global variables
	global stemmedStopwords
	#create lower version of text
	ltext = text.lower()
	#initialize empty list of extra-features
	efeatures = []
	#1) count urls
	efeatures.append(float(len([ t for t in tokens if t==tokenizer.URL ])))
	#2) count mentions
	efeatures.append(float(len([ t for t in tokens if t==tokenizer.ATHANDLE ])))
	#3) is a reply (tweet)
	efeatures.append(1. if len(text)>1 and text[0]=='@' else 0.)
	#4) count hashtags
	lht = re_ht.findall(text)
	efeatures.append(float(len(lht)))
	#5) max length of an hashtags
	mht = 0
	for ht in lht:
		if len(ht)>mht: mht = len(ht)
	efeatures.append(float(mht))
	#6) is a retweet rt
	efeatures.append(1. if len(ltext)>=4 and ltext[:3]=='rt ' else 0.)
	#7) percentage of capital letters
	efeatures.append(capitalLetters(text))
	#8) percentage of all-capital words
	words = re_words.findall(text)
	ncw = len([ w for w in words if w==w.upper() ])
	efeatures.append(ncw/len(words) if len(words)>0 else 0.)
	#9) count hate words
	efeatures.append(float(len([ t for t in tokens if t in hatesv ])))
	#10) count exclamation mark !
	efeatures.append(float(len([ c for c in text if c=='!' ])))
	#11) count dots
	efeatures.append(float(len(re_dots.findall(text)) + len([c for c in text if c=='…'])))
	#12) count punctuations chars
	efeatures.append(float(len([ c for c in text if c in tokenizer.punctuation ])))
	#13) count emojis
	efeatures.append(float(len([ t for t in tokens if len(t)>0 and (t[0] in tokenizer.emojiChars or t[0] in tokenizer.punctuation) ])))
	#14) count multiple vowels
	nmv = 0
	for s in tokenizer.many_a.findall(ltext): nmv += len(s)
	for s in tokenizer.many_e.findall(ltext): nmv += len(s)
	for s in tokenizer.many_i.findall(ltext): nmv += len(s)
	for s in tokenizer.many_o.findall(ltext): nmv += len(s)
	for s in tokenizer.many_u.findall(ltext): nmv += len(s)
	efeatures.append(float(nmv))
	#15) percentage of correct words
	ncw = len([ w for w in words if speller.check(w.lower()) or (len(w)>0 and w[0]==w[0].upper() and w[1:]==w[1:].lower()) ])
	efeatures.append(ncw/len(words) if len(words)>0 else 0.)
	#16-17) sentiment polarity and intensity
	polar,inten = 0.,0.
	for t in tokens:
		if t in sentix:
			polar += sentix[t][0]
			inten += sentix[t][1]
	polar /= len(tokens)
	inten /= len(tokens)
	efeatures.append(polar)
	efeatures.append(inten)
	#18) length in chars
	efeatures.append(float(len(text)))
	#19) length in tokens
	efeatures.append(float(len(tokens) if len(tokens)>0 and tokens[0]!=tokenizer.EMPTY else 0))	
	#20) count stopwords
	if stemmedStopwords is None:
		stemmedStopwords = [ tokenizer.stem(w) for w in tokenizer.stopwordsList ] if stemming else tokenizer.stopwordsList
	efeatures.append(float(len([ t for t in tokens if t in stemmedStopwords ])))
	#return efeatures converted to a numpy array
	return np.array(efeatures,dtype=dtype)


#return #capital_letters/#total_letters (if no letter, return 0)
def capitalLetters(text):
	cl = 0
	tl = 0
	for c in text:
		if c.isalpha():
			tl += 1
			if c==c.upper(): cl += 1
	return cl/tl if tl>0 else 0.


def usage():
	print('USAGE: python3 featuresGenerator.py INPUT_FILE MODEL_FILE OUTPUT_FILE [-verbose y|n] [-extraf y|n] [-hatesv FILENAME] [-sentix FILENAME] [-stemming y|n] [-stopwords y|n] [-idf y|n] [-favg y|n] [-fstd y|n] [-fmin y|n] [-fmax y|n] [-fmed y|n] [-fsum y|n] [-fdavg y|n] [-fdstd y|n] [-fdmin y|n] [-fdmax y|n] [-fdmed y|n] [-fdsum y|n] [-allfeat y|n] [-allstat y|n] [-alldstat y|n]',file=sys.stderr)
	print('DEFAULT:',file=sys.stderr)
	print('\t-verbose   y',file=sys.stderr)
	print('\t-extraf    y',file=sys.stderr)
	print('\t-hatesv    original/hate_speech_vocabulary.txt',file=sys.stderr)
	print('\t-sentix    original/sentix',file=sys.stderr)
	print('\t-stemming  y',file=sys.stderr)
	print('\t-stopwords y',file=sys.stderr)
	print('\t-idf       n',file=sys.stderr)
	print('\t-favg      y',file=sys.stderr)
	print('\t-fstd      n',file=sys.stderr)
	print('\t-fmin      n',file=sys.stderr)
	print('\t-fmax      n',file=sys.stderr)
	print('\t-fmed      n',file=sys.stderr)
	print('\t-fsum      n',file=sys.stderr)
	print('\t-fdavg     n',file=sys.stderr)
	print('\t-fdstd     n',file=sys.stderr)
	print('\t-fdmin     n',file=sys.stderr)
	print('\t-fdmax     n',file=sys.stderr)
	print('\t-fdmed     n',file=sys.stderr)
	print('\t-fdsum     n',file=sys.stderr)
	print('\t-allfeat   n',file=sys.stderr)
	print('\t-allstat   n',file=sys.stderr)
	print('\t-alldstat  n',file=sys.stderr)
	print('NOTE_1: it can read utf-8 and latin_1 (ISO-8859) encodings.',file=sys.stderr)


#main part
if __name__ == '__main__':
	if len(sys.argv)<4:
		usage()
		sys.exit()
	inputFilename = sys.argv[1]
	wvModelFilename = sys.argv[2]
	outputFilename = sys.argv[3]
	verbose = True
	extraf = True
	hatesvFilename = 'original/hate_speech_vocabulary.txt'
	sentixFilename = 'original/sentix'
	stemming = True
	stopwords = True
	idf = False
	favg = True
	fstd = False
	fmin = False
	fmax = False
	fmed = False
	fsum = False
	fdavg = False
	fdstd = False
	fdmin = False
	fdmax = False
	fdmed = False
	fdsum = False
	i = 4
	while i<len(sys.argv):
		if sys.argv[i] in {'-verbose', '-v'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -verbose requires y|n argument")
			verbose = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-extraf', '-extra', '-extrafeat'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -extraf requires y|n argument")
			extraf = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-hatesv', '-hate'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -hatesv requires FILENAME argument")
			hatesvFilename = sys.argv[i+1]
		elif sys.argv[i] in {'-sentix', '-sent'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -sentix requires FILENAME argument")
			sentixFilename = sys.argv[i+1]
		elif sys.argv[i] in {'-stemming', '-stem'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stemming requires y|n argument")
			stemming = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-stopwords', '-sw'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stopwords requires y|n argument")
			stopwords = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i]=='-idf':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -idf requires y|n argument")
			idf = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-favg', '-avg'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -favg requires y|n argument")
			favg = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fstd', '-std'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fstd requires y|n argument")
			fstd = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fmin', '-min'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fmin requires y|n argument")
			fmin = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fmax', '-max'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fmax requires y|n argument")
			fmax = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fmed', '-med'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fmed requires y|n argument")
			fmed = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fsum', '-sum'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fsum requires y|n argument")
			fsum = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdavg', '-davg'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdavg requires y|n argument")
			fdavg = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdstd', '-dstd'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdstd requires y|n argument")
			fdstd = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdmin', '-dmin'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdmin requires y|n argument")
			fdmin = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdmax', '-dmax'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdmax requires y|n argument")
			fdmax = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdmed', '-dmed'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdmed requires y|n argument")
			fdmed = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-fdsum', '-dsum'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -fdsum requires y|n argument")
			fdsum = sys.argv[i+1][0] not in 'nNfF0'
		elif sys.argv[i] in {'-allfeat', '-all', '-fall'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -allfeat requires y|n argument")
			if sys.argv[i+1][0] not in 'nNfF0':
				favg=fstd=fmin=fmax=fmed=fsum=True
				fdavg=fdstd=fdmin=fdmax=fdmed=fdsum=True
				extraf=True
		elif sys.argv[i] == '-allstat':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -allstat requires y|n argument")
			if sys.argv[i+1][0] not in 'nNfF0': favg=fstd=fmin=fmax=fmed=fsum=True
		elif sys.argv[i] == '-alldstat':
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -alldstat requires y|n argument")
			if sys.argv[i+1][0] not in 'nNfF0': fdavg=fdstd=fdmin=fdmax=fdmed=fdsum=True
		i += 2
	postsWithFeatures = generateFeatures(inputFilename,wvModelFilename,hatesvFilename,sentixFilename,
							stemming,stopwords,
							favg,fstd,fmin,fmax,fmed,fsum,
							fdavg,fdstd,fdmin,fdmax,fdmed,fdsum,
							idf,extraf,verbose)
	if verbose: print('* Saving the features in '+outputFilename+' ...')
	saveFeatures(postsWithFeatures,outputFilename)
	if verbose: print('  Features saved in '+outputFilename+'.')

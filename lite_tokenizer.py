#these lines are required to not print a "benign" warning
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import nltk.tokenize
import nltk.stem.snowball
import splitter
import emoji
import re
import sys
import time
#import aspell #debug
import util


#tokens for normalized numbers and urls
NUMBER   = '<NUM>'
URL      = '<URL>'
ATHANDLE = '<ATH>'
EMPTY    = '<EMP>'

#Create a NLTK TweetTokenizer instance that: convert to lower case, reduce more than 3 equal consecutive letters to 3, do not strip @username tokens
tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
#Create an ASPELL speller for italian
#speller = aspell.Speller('lang','it') #debug
#Create a NLTK Snowball stemmer for italian
stemmer = nltk.stem.snowball.ItalianStemmer(ignore_stopwords=False)

#Some compiled regular expression
whitespaces = re.compile('\s+')
macApostroph = re.compile('[’´‘]')
leftRightDoubleQuotations = re.compile('[“”]')
bom = re.compile('\ufeff') #byte order mark, or BOM
many_a = re.compile('aa+')
many_e = re.compile('ee+')
many_i = re.compile('ii+')
many_o = re.compile('oo+')
many_u = re.compile('uu+')
coo = re.compile('^coo.*')
app_acc_a = re.compile("a'")
app_acc_e = re.compile("e'")
app_acc_i = re.compile("i'")
app_acc_o = re.compile("o'")
app_acc_u = re.compile("u'")
url = re.compile('^http.*$')
number = re.compile('^-?[0-9]+[\.\,\:/\\-]?[0-9]*$') #numbers, timings, dates, fractions
underscores = re.compile('_+')
normalizationMark = re.compile('^<[A-Z]{3}>$')
emojiModifier = re.compile('^[\U0001f3fb-\U0001f3ff]$')

#Some sets of strings
percheSpellings = { 'xkè', 'xké', 'xke', 'xché', 'xchè', 'xche', 'perkè', 'perké', 'perke', 'perchè', 'perche' }
cazzoSpellings  = { 'caxxo', 'caxxi', 'c@zzo', 'c@zzi', 'casso', 'cassi', 'cazo', 'cazi', 'cazzi', 'azz' }
vaffaSpellings  = { 'vaf', 'vaff', 'vaffa', 'affanculo', 'afanculo', 'fanculo', 'vafanculo' }
punctuation = { '‘', '’', '´', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '–', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '...', '..', '…', '°', '«', '»' }
dots = { '.', '..', '...' }
emojiChars = emoji.UNICODE_EMOJI.keys()
stopwordsList = ['ad', 'al', 'allo', 'ai', 'agli', 'all', 'agl', 'alla', 'alle', 'con', 'col', 'coi', 'da', 'dal', 'dallo', 'dai', 'dagli', 'dall', 'dagl', 'dalla', 'dalle', 'di', 'del', 'dello', 'dei', 'degli', 'dell', 'degl', 'della', 'delle', 'in', 'nel', 'nello', 'nei', 'negli', 'nell', 'negl', 'nella', 'nelle', 'su', 'sul', 'sullo', 'sui', 'sugli', 'sull', 'sugl', 'sulla', 'sulle', 'per', 'tra', 'contro', 'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue', 'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre', 'vostro', 'vostra', 'vostri', 'vostre', 'mi', 'ti', 'ci', 'vi', 'lo', 'la', 'li', 'le', 'gli', 'ne', 'il', 'un', 'uno', 'una', 'ma', 'ed', 'se', 'perché', 'anche', 'come', 'dov', 'dove', 'che', 'chi', 'cui', 'non', 'più', 'quale', 'quanto', 'quanti', 'quanta', 'quante', 'quello', 'quelli', 'quella', 'quelle', 'questo', 'questi', 'questa', 'queste', 'si', 'tutto', 'tutti', 'a', 'c', 'e', 'i', 'l', 'o', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno', 'abbia', 'abbiate', 'abbiano', 'avrò', 'avrai', 'avrà', 'avremo', 'avrete', 'avranno', 'avrei', 'avresti', 'avrebbe', 'avremmo', 'avreste', 'avrebbero', 'avevo', 'avevi', 'aveva', 'avevamo', 'avevate', 'avevano', 'ebbi', 'avesti', 'ebbe', 'avemmo', 'aveste', 'ebbero', 'avessi', 'avesse', 'avessimo', 'avessero', 'avendo', 'avuto', 'avuta', 'avuti', 'avute', 'sono', 'sei', 'è', 'siamo', 'siete', 'sia', 'siate', 'siano', 'sarò', 'sarai', 'sarà', 'saremo', 'sarete', 'saranno', 'sarei', 'saresti', 'sarebbe', 'saremmo', 'sareste', 'sarebbero', 'ero', 'eri', 'era', 'eravamo', 'eravate', 'erano', 'fui', 'fosti', 'fu', 'fummo', 'foste', 'furono', 'fossi', 'fosse', 'fossimo', 'fossero', 'essendo', 'faccio', 'fai', 'facciamo', 'fanno', 'faccia', 'facciate', 'facciano', 'farò', 'farai', 'farà', 'faremo', 'farete', 'faranno', 'farei', 'faresti', 'farebbe', 'faremmo', 'fareste', 'farebbero', 'facevo', 'facevi', 'faceva', 'facevamo', 'facevate', 'facevano', 'feci', 'facesti', 'fece', 'facemmo', 'faceste', 'fecero', 'facessi', 'facesse', 'facessimo', 'facessero', 'facendo', 'sto', 'stai', 'sta', 'stiamo', 'stanno', 'stia', 'stiate', 'stiano', 'starò', 'starai', 'starà', 'staremo', 'starete', 'staranno', 'starei', 'staresti', 'starebbe', 'staremmo', 'stareste', 'starebbero', 'stavo', 'stavi', 'stava', 'stavamo', 'stavate', 'stavano', 'stetti', 'stesti', 'stette', 'stemmo', 'steste', 'stettero', 'stessi', 'stesse', 'stessimo', 'stessero', 'stando'] #obtained with nltk library but hardcoded here
stopwordsList = stopwordsList + ['gl','gliel'] #added by myself


#main function of this module: takes in input a post (tweet or fb post) and returns a list of tokens
def tokenize(post,stemming=True,stopwords=True):
	#compute the tokens list by transforming the list produced NLTK TweetTokenizer on the transformed post, then transform each token and filter out some tokens
	tokens = [ stem(t) if stemming else t for t in map(transformToken,transformTokensList(tokenizer.tokenize(transformPost(post)))) if preserveToken(t,stopwords=stopwords) ]
	if len(tokens)==0: tokens = [ EMPTY ]
	return tokens


def transformTokensList(tokens):
	#init empty result list
	trTokens = []
	#scan the original list and build-up the merged one
	n = len(tokens)
	i = 0
	while i<n:
		#merge emoji to emoji-modifiers
		if tokens[i] in emojiChars:
			t = tokens[i]
			j = i
			while j<n:
				j += 1
				if j<n and emojiModifier.match(tokens[j]) is not None:
					t += tokens[j]
				else:
					break
			trTokens.append(t)
			i += j-i
		#split tokens containing apostroph
		elif "'" in tokens[i]:
			l = tokens[i].split("'")
			if len(l)>2 or (len(l)>1 and (len(l[1])>0 and l[1][0] not in 'aeiouhè')):
				trTokens.append(tokens[i])
			elif len(l)==2:
				trTokens.extend(l)
			i += 1
		#split tokens containing a dot
		elif '.' in tokens[i]:
			trTokens.extend(tokens[i].split('.'))
			i += 1
		#split tokens containing one or more underscores
		elif underscores.search(tokens[i]) is not None:
			trTokens.extend(underscores.split(tokens[i]))
			i += 1
		#normal
		else:
			trTokens.append(tokens[i])
			i += 1
	#return the produced list
	return trTokens

#transform the text of the post
def transformPost(post):
	#init transformed post to the original post
	trPost = post
	#remove BOM (byte order mark) characters
	trPost = bom.sub('',trPost)
	#replace Mac apostrophes with normal apostrophes
	trPost = macApostroph.sub("'",trPost)
	#replace left/right double quotations with normal ones
	trPost = leftRightDoubleQuotations.sub('"',trPost)
	#correct apostrophes used as accents
	trPost = apostrophesToAccents(trPost)
	#squeeze consecutive, heading and trailing whitespaces 
	trPost = whitespaces.sub(' ',trPost).strip()
	#return transformed post
	return trPost

	
#transform the text of a single token
def transformToken(token):
	#init transformed token to the original token
	trToken = token
	#normalize numbers
	trToken = number.sub(NUMBER,trToken)
	#normalize urls
	trToken = url.sub(URL,trToken)
	#normalize at_handles
	if len(trToken)>0 and trToken[0]=='@': trToken = ATHANDLE
	#squeeze vowels
	trToken = squeezeVowels(trToken)
	#remove -
	trToken = trToken.replace('-','')
	#return the transformed token
	return trToken


#return true if the token has to be preserved, false otherwise
def preserveToken(token,stopwords):
	#remove empty tokens (appearing for whatever reason)
	if len(token)==0: return False
	#remove twitter rt (RT) tokens
	if token=='rt': return False
	#remove punctuation tokens
	if (len(token)==1 and not token.isalnum() and not token in emojiChars and emojiModifier.match(token) is None) or token in punctuation: return False
	#remove "strange" tokens containing only spaces and dots
	if set(token)<={' ','.'}: return False
	#remove stopwords
	if not stopwords and token in stopwordsList: return False
	#if not returned before, it is ok
	return True
	#return not speller.check(token) #debug


def stem(token):
	#stem the token (if it isnt a normalization mark)
	return stemmer.stem(token) if normalizationMark.match(token) is None else token


#correct apostrophes used as accents
def apostrophesToAccents(text):
	text = app_acc_a.sub('à',text)
	text = app_acc_e.sub('è',text) #if é will be verified and corrected in the token
	text = app_acc_i.sub('ì',text)
	text = app_acc_o.sub('ò',text)
	text = app_acc_u.sub('ù',text)
	return text


#squueze consecutive equal vowels
def squeezeVowels(token):
	token = many_a.sub('a',token)
	token = many_e.sub('e',token)
	token = many_i.sub('i',token)
	if coo.match(token) is None: token = many_o.sub('o',token)
	token = many_u.sub('u',token)
	return token


#correct accents on eéè and also the k in ch
def correctEaccents(token):
	if len(token)==1 and token=='é':
		return 'è'
	elif token=='ke':
		return 'che'
	elif len(token)>3 and token[-3:]=='chè':
		return token[:-3]+'ché'
	elif len(token)>2 and (token[-2:]=='ké' or token[-2:]=='kè'):
		return token[:-2]+'ché'
	elif len(token)>2 and  token[-2:]=='ke':
		return token[:-2]+'che'
	return token
		

#normalize laughs
def normalizeLaughs(token):
	if token!='ha' and ( {'a','h'}<=set(token)<={'a','h','g'} or {'e','h'}<=set(token)<={'e','h','g'} ):
		return 'ahahah'
	return token


#print instructions for command line
def usage():
	print('USAGE: python3 tokenizer.py INPUT_FILE OUTPUT_FILE [-stemming y|n] [-stopwords y|n] [-tsvout y|n] [-verbose y|n] [-from_line <NUM>] [-to_line <NUM>|LAST]',file=sys.stderr)
	print('DEFAULT:',file=sys.stderr)
	print('\t-stemming  y',file=sys.stderr)
	print('\t-stopwords y',file=sys.stderr)
	print('\t-tsvout    y (meaningful only if INPUT_FILE is .tsv)',file=sys.stderr)
	print('\t-verbose   y',file=sys.stderr)
	print('\t-from_line 1',file=sys.stderr)
	print('\t-to_line   LAST',file=sys.stderr)
	print('NOTE_1: it can read utf-8 and latin_1 (ISO-8859) encodings, but the output is only utf-8.',file=sys.stderr)
	print('NOTE_2: INPUT_FILE can be set to "stdin" (or "-"). OUTPUT_FILE can be set to "stdout" (or "-"). When OUTPUT_FILE is "stdout", the verbose printings go to the stderr.',file=sys.stderr)


#main part of the module
if __name__ == '__main__':
	if len(sys.argv)<3:
		usage()
		sys.exit()
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
	stdin = inputFile in {'stdin','-'}
	stdout = outputFile in {'stdout','-'}
	tsvin = inputFile[-4:]=='.tsv'
	tsvout = True
	stemming = True
	stopwords = True
	verbose = True
	from_line = 1
	to_line = util.BIGINT
	i = 3
	while i<len(sys.argv):
		if sys.argv[i] in {'-stemming', '-stem'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stemming requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': stemming = False
		elif sys.argv[i] in {'-stopwords', '-sw'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -stopwords requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': stopwords = False
		elif sys.argv[i] in {'-tsvout', '-outtsv', '-tsv'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -tsvout requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': tsvout = False
		elif sys.argv[i] in {'-verbose', '-v'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -verbose requires y|n argument")
			if sys.argv[i+1][0] in 'nNfF0': verbose = False
		elif sys.argv[i] in {'-from_line', '-from'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -from_line requires <NUM> argument")
			from_line = int(sys.argv[i+1])
		elif sys.argv[i] in {'-to_line', '-to'}:
			if i+1>=len(sys.argv) or sys.argv[i+1][0]=='-': sys.exit("ERROR: -to_line requires <NUM>|LAST argument")
			to_line = util.BIGINT if sys.argv[i+1] in {'LAST','last','L','l'} else int(sys.argv[i+1])
		i += 2
	t0 = time.process_time()
	fin = sys.stdin if stdin else open(inputFile,'r',encoding=util.getFileEncoding(inputFile))
	fout = sys.stdout if stdout else open(outputFile,'w',encoding='utf-8')
	for i,id,post,label in util.scanFileLines(fin,tsvin,from_line=from_line,to_line=to_line):
		id,label = (id+'\t','\t'+label) if tsvin and tsvout else ('','')
		print(id+(' '.join(tokenize(post,stemming,stopwords)))+label,file=fout)
		if verbose: print('\r'+str(i-from_line+1),end=' lines tokenized ...',file=(sys.stderr if stdout else sys.stdout))
	if not stdin: fin.close()
	if not stdout: fout.close()
	if verbose: print('\nDone in '+str(round(time.process_time()-t0,3))+' seconds. Tokens have been saved in '+outputFile,file=(sys.stderr if stdout else sys.stdout))

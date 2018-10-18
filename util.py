import magic


#integer +infinity
BIGINT = 1000000000 #1bilion


#Scan a file yielding one-by-one the 4-ples <i,id,post,label> where:
# i     = int counter of the line
# id    = id string of the .tsv file format, '' if tsv==False
# post  = sentence string
# label = label string of the .tsv file format, '' if tsv==False
def scanFileLines(f,tsv=False,from_line=1,to_line=BIGINT):
	i = 0
	for line in f:
		i += 1
		if i<from_line: continue
		id,post,label = line.strip().split('\t') if tsv else ('',line.strip(),'')
		yield (i,id,post,label)
		if i>=to_line: break


#Guess the encoding of a file. It supports: latin_1 (ISO-8859) and utf-8
def getFileEncoding(filename):
	return 'latin_1' if magic.from_file(filename)=='ISO-8859 text' else 'utf-8'

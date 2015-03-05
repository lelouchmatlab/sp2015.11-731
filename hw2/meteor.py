#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import re
# DRY


        
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!

def match(w1, w2):
    #stemmer = SnowballStemmer("english")
    #w1=re.sub(r'[\x00-\xff]','',w1)
    #w2=re.sub(r'[\x00-\xff]','',w2)
    #return stemmer.stem(w1)==stemmer.stem(w2)
    return w1==w2
def meteor_score(h, ref):
    for i in xrange(len(h)):
        h[i]=h[i].lower()
    for i in xrange(len(ref)):
        ref[i]=ref[i].lower()
    #print "CURRENT PAIR: "
    #print h
    #print ref
    len_h = len(h)
    len_ref = len(ref)
    h_buf = []
    ref_buf = []
    ref_index = set()
    common = 0
    for h_word in h:
        for i in xrange(len(ref)):
            if (i not in ref_index) and match(ref[i],h_word):
	        h_buf.append(h_word)
	        common += 1.0  
	        ref_index.add(i)
	        break 
    ref_index_list = list(ref_index)
    ref_index_list.sort()
    for index in ref_index_list:
        ref_buf.append(ref[index])
    
    frag = 0
    while ref_index:
        index = ref_index.pop()
        pointer = index - 1
        while pointer in ref_index:
	    ref_index.remove(pointer)
	    pointer = pointer -1
	pointer = index + 1
	while pointer in ref_index:
	    ref_index.remove(pointer)
	    pointer = pointer +1
	frag = frag + 1.0
    
    precison = common/len_h
    recall = common/len_ref
    if precison ==0:
        return 0
    Fmean = (10*precison*recall)/(9*precison+recall)
    #print h_buf
    #print ref_buf
    if len(h_buf) == 1:
        frag = 1
    else:
        frag = (frag-1)/(len(h_buf)-1)
    DF = 0.5*(pow(frag,3))
    
    score = (1-DF)*Fmean
    """
    print "Precison: %f"% precison
    print "Recall: %f"%recall
    print "Frag: %f"%frag
    print "DF: %f"%DF
    print "Score: %f"%score
    print h_buf
    print ref_buf
    """
    return score
      
        
    
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        tokenizer = RegexpTokenizer(r'\w+')
        with open(opts.input) as f:
            for pair in f:
	        #yield [tokenizer.tokenize(sentence.replace('"',' ')) for sentence in pair.split(' ||| ')]
                yield [sentence.replace('"'," ").strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    
    #ref='the Iraqi\'s weapons are to be handed over to the army within two weeks'.lower().split()
    #h='in two weeks Iraqi weapons will give army'.lower().split()
    #meteor_score(h,ref)
    
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        
        h1_score = meteor_score(h1,ref)
        h2_score = meteor_score(h2,ref)
         
        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}
          
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=1, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
opts = parser.parse_args()

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

hypothesis = namedtuple('hypothesis', 'logprob, lm_state, predecessor,predecessor_end, current_start, current_end,  phrase')
for f in input_sents:
    # The following code implements a DP monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of 
    # the first i words of the input sentence.
    # HINT: Generalize this so that stacks[i] contains translations
    # of any i words (remember to keep track of which words those
    # are, and to estimate future costs)
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, -1, 0, 0, None)
    #print "Source sentence: ",f
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        # extend the top s hypotheses in the current stack
        #print "\n",i, " --------------------------------\n\n"
        #print "/////////////////////////////////////////////"
        #for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h:h.logprob):
            #print h.predecessor_end, h.current_start, h.current_end,h.lm_state,h.logprob
        #print "////////////////////////////////////////////"
        #print h.predecessor_end, h.current_start, h.current_end,h.lm_state,h.logprob, '**********'
        for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h: h.logprob):
            if h.predecessor_end + 1 == h.current_start:
                for j in xrange(i+1,len(f)+1):
                    if f[i:j] in tm:
                        #print f[i:j], '######## MONO ########'
                        for phrase in tm[f[i:j]]:
                            #print phrase.logprob,
                            logprob = h.logprob + phrase.logprob
                            lm_state = h.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                #print word, word_logprob
                                logprob += word_logprob
                            logprob += lm.end(lm_state) if j == len(f) else 0.0
                            #print logprob
                            new_hypothesis = hypothesis(logprob,lm_state, h, i-1, i, j-1, phrase)
                            #print j, new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start,new_hypothesis.current_end
                            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                                stacks[j][lm_state] = new_hypothesis
                                #print j, new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start, new_hypothesis.current_end
                        for k in xrange(j+1,len(f)+1):
                            if f[j:k] in tm:
                                #print f[j:k], '$$$$$$$$ SWAP $$$$$$$$'
                                for phrase in tm[f[j:k]]:
                                    logprob = h.logprob + phrase.logprob
                                    lm_state = h.lm_state
                                    for word in phrase.english.split():
                                        (lm_state, word_logprob) = lm.score(lm_state,word)
                                        logprob += word_logprob
                                    #logprob += lm.end(lm_state) if k == len(f) else 0.0
                                    new_hypothesis = hypothesis(logprob,lm_state, h,i-1,j, k-1, phrase)
                                    #print i+k-j,new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start,new_hypothesis.current_end
                                    if lm_state not in stacks[i+k-j] or stacks[i+k-j][lm_state].logprob < logprob:
                                        stacks[i+k-j][lm_state] = new_hypothesis
                                        #print i+k-j, new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start,new_hypothesis.current_end
            else:
                jump_back = f[h.predecessor_end+1:h.current_start]
                #print h.lm_state, h.predecessor_end+1,h.current_start,jump_back,'******** jump_back ********'
                for phrase in tm[jump_back]:
                    logprob = h.logprob + phrase.logprob
                    lm_state = h.lm_state
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob
                    logprob += lm.end(lm_state) if h.current_end == len(f)-1 else 0.0
                    new_hypothesis = hypothesis(logprob,lm_state, h, h.predecessor_end, h.predecessor_end+1,h.current_end, phrase)
                    #print h.current_end+1,new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start,new_hypothesis.current_end
                    if lm_state not in stacks[h.current_end+1] or stacks[h.current_end+1][lm_state].logprob < logprob:
                        stacks[h.current_end+1][lm_state] = new_hypothesis
                        #print h.current_end+1, new_hypothesis.logprob,new_hypothesis.lm_state,new_hypothesis.predecessor_end,new_hypothesis.current_start,new_hypothesis.current_end

    # find best translation by looking at the best scoring hypothesis
    # on the last stack
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    def extract_english_recursive(h):
        return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)
    print extract_english_recursive(winner)
    def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    for ele in stacks[-1].itervalues():
        tm_logprob = extract_tm_logprob(ele)
        #print extract_english_recursive(ele),tm_logprob, ele.logprob - tm_logprob, ele.logprob



    if opts.verbose:
        #def extract_tm_logprob(h):
        #    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write('LM = %f, TM = %f, Total = %f\n' % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
        #for ele in stacks[-1].itervalues():
        #    tm_logprob = extract_tm_logprob(ele)
        #    sys.stderr.write('LM = %f, TM = %f, Total = %f\n' %(ele.logprob -
        #        tm_logprob, tm_logprob, ele.logprob))
    




#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=20, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
opts = parser.parse_args()

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]


def score_lm(translation,isEnd):
    lm_score = 0.0
    lm_state = lm.begin()
    for word in ' '.join(translation).split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        lm_score += word_logprob
    lm_score += lm.end(lm_state) if isEnd is True else 0.0
    return lm_score,lm_state
    
    
hypothesis = namedtuple('hypothesis', 'logprob, tm_score, lm_score, lm_state, translation')
for f in input_sents:
    # The following code implements a DP monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of 
    # the first i words of the input sentence.
    # HINT: Generalize this so that stacks[i] contains translations
    # of any i words (remember to keep track of which words those
    # are, and to estimate future costs)
    initial_hypothesis = hypothesis(0.0, 0.0, 0.0, lm.begin(), [])

    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        # extend the top s hypotheses in the current stack
        for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h: h.logprob): # prune
            for j in xrange(i+1,len(f)+1):
                if f[i:j] in tm:
                    for phrase in tm[f[i:j]]:
                        tm_score = h.tm_score + phrase.logprob
                        for k in xrange( max(0, len(h.translation)-2), len(h.translation) +1):
                            new_translation = h.translation[:k] + [phrase.english] + h.translation[k:]
                            isEnd = False
                            if j==len(f):
                                isEnd = True
                            lm_score,lm_state = score_lm(new_translation, isEnd)
                            new_hypothesis = hypothesis(tm_score + lm_score, tm_score, lm_score, lm_state, new_translation)
                            if new_hypothesis.lm_state not in stacks[j] or stacks[j][new_hypothesis.lm_state].logprob < new_hypothesis.logprob: # second case is recombination
                                stacks[j][new_hypothesis.lm_state] = new_hypothesis
    # find best translation by looking at the best scoring hypothesis
    # on the last stack
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    def extract_english_recursive(h):
        return ' '.join(h.translation)
    print extract_english_recursive(winner)

    
    
    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write('LM = %f, TM = %f, Total = %f\n' % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

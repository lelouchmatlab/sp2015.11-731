import optparse
import sys
from collections import defaultdict
import random
#import pickle

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--num_iteration", dest="num_iter", default=6, type="int", help="Number of iterations in IBM 1")
(opts, _) = optparser.parse_args()


train_text = [[sentence.lower().strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
test_text=[[sentence.lower().strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:150]


#f=lambda :random.random()/10
f=lambda :1.0/10
Prob_eg=defaultdict(lambda:defaultdict(f))   #P[g_w][e_w]
sys.stderr.write("IBM 1....\n")
for i in xrange(opts.num_iter):
  sys.stderr.write("Iter\n")
  Statis_eg=defaultdict(lambda:defaultdict(float))
  for(n,(g,e)) in enumerate(train_text):
    if n % 500 == 0:
      sys.stderr.write(".")
    for target_word_e in e:
      denominator=0
      for source_word_g in g:
	denominator=denominator+Prob_eg[source_word_g][target_word_e]
      for source_word_g in g:
	delta=Prob_eg[source_word_g][target_word_e]/denominator
	Statis_eg[source_word_g][target_word_e]=Statis_eg[source_word_g][target_word_e]+delta
  
  for source_word_g in Prob_eg.keys():
    normalization=0
    for target_word_e in Prob_eg[source_word_g].keys():
      normalization=normalization+Statis_eg[source_word_g][target_word_e]
    for target_word_e in Prob_eg[source_word_g].keys():
      Prob_eg[source_word_g][target_word_e]=Statis_eg[source_word_g][target_word_e]/normalization
  sys.stderr.write("\n") 


  #with open('IBM1.pickle','wb') as handle:
    #pickle.dump(Prob_eg,handle)  
"""
for(g,e) in train_text:
  for (e_idx, target_word_e) in enumerate(e):
    best_idx=0
    score=0
    for (g_idx, source_word_g) in enumerate(g):
      if Prob_eg[source_word_g][target_word_e]>score:
	best_idx=g_idx
	score=Prob_eg[source_word_g][target_word_e]
    sys.stdout.write("%i-%i " % (best_idx,e_idx))
  sys.stdout.write("\n") 
"""


Prob_ali=defaultdict(lambda:defaultdict(f))
sys.stderr.write("IBM 2....\n")
num_iter=16
for i in xrange(num_iter):
  sys.stderr.write("Iter\n")
  Statis_ali=defaultdict(lambda:defaultdict(float))
  Statis_eg=defaultdict(lambda:defaultdict(float))
  for(n,(g,e)) in enumerate(train_text):
    if n % 500 == 0:
      sys.stderr.write(".")
    for (e_idx,target_word_e) in enumerate(e):
      denominator=0
      for (g_idx,source_word_g) in enumerate(g):
	denominator=denominator+Prob_eg[source_word_g][target_word_e]*Prob_ali[(e_idx,len(g),len(e))][g_idx]
      for (g_idx,source_word_g) in enumerate(g):
	delta=Prob_ali[(e_idx,len(g),len(e))][g_idx]*Prob_eg[source_word_g][target_word_e]/denominator
	Statis_eg[source_word_g][target_word_e]=Statis_eg[source_word_g][target_word_e]+delta
	Statis_ali[(e_idx,len(g),len(e))][g_idx]=Statis_ali[(e_idx,len(g),len(e))][g_idx]+delta
  
  for source_word_g in Prob_eg.keys():
    normalization=0
    for target_word_e in Prob_eg[source_word_g].keys():
      normalization=normalization+Statis_eg[source_word_g][target_word_e]
    for target_word_e in Prob_eg[source_word_g].keys():
      Prob_eg[source_word_g][target_word_e]=Statis_eg[source_word_g][target_word_e]/normalization
  for (e_idx,len_g,len_e) in Prob_ali.keys():
    normalization=0
    for g_idx in Prob_ali[(e_idx,len_g,len_e)].keys():
      normalization=normalization+Statis_ali[(e_idx,len_g,len_e)][g_idx]
    for g_idx in Prob_ali[(e_idx,len_g,len_e)].keys():
      Prob_ali[(e_idx,len_g,len_e)][g_idx]=Statis_ali[(e_idx,len_g,len_e)][g_idx]/normalization
  sys.stderr.write("\n") 
  
  #with open('IBM2.pickle','wb') as handle:
    #pickle.dump((Prob_eg,Prob_ali),handle)
for(g,e) in train_text:
  for (e_idx, target_word_e) in enumerate(e):
    best_idx=0
    score=0
    for (g_idx, source_word_g) in enumerate(g):
      if Prob_ali[(e_idx,len(g),len(e))][g_idx]*Prob_eg[source_word_g][target_word_e]>score:
	best_idx=g_idx
	score=Prob_ali[(e_idx,len(g),len(e))][g_idx]*Prob_eg[source_word_g][target_word_e]
    sys.stdout.write("%i-%i " % (best_idx,e_idx))
  sys.stdout.write("\n") 


      



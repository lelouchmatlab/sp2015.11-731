import optparse
import sys
from collections import defaultdict
import random

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--num_iteration", dest="num_iter", default=6, type="int", help="Number of iterations in IBM 1")
(opts, _) = optparser.parse_args()


train_text = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
test_text=[[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:150]


#f=lambda :random.random()/10
f=lambda :1.0/10
Prob_eg=defaultdict(lambda:defaultdict(f))   #P[g_w][e_w]
#c_g=defaultdict(float)


#vocab_g=defaultdict(set)
#for (n,(g,e)) in enumerate(train_text):
#  for g_word in g:
#    for e_word in e:
#      vocab_g[g_word].add(e_word)
      
#for g_word in vocab_g.keys():
#  length=len(vocab_g[g_word])
#  for e_word in vocab_g[g_word]:
#    t_e_g[e_word][g_word]=1.0/length


for i in xrange(opts.num_iter):
  
  Statis_eg=defaultdict(lambda:defaultdict(float))
  for(n,(g,e)) in enumerate(train_text):
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


q=defaultdict(f)

num_iter=20
for i in xrange(num_iter):
  c_s_t=defaultdict(float)
  c_t=defaultdict(float)
  c_e_g=defaultdict(lambda:defaultdict(float))
  c_g=defaultdict(float)
  for(n,(g,e)) in enumerate(train_text):
    for (e_idx,target_word_e) in enumerate(e):
      denominator=0
      for (g_idx,source_word_g) in enumerate(g):
	denominator=denominator+t_e_g[target_word_e][source_word_g]*q[(g_idx,e_idx,len(g),len(e))]
      for (g_idx,source_word_g) in enumerate(g):
	delta=q[(g_idx,e_idx,len(g),len(e))]*t_e_g[target_word_e][source_word_g]/denominator
	c_e_g[target_word_e][source_word_g]=c_e_g[target_word_e][source_word_g]+delta
	c_g[source_word_g]=c_g[source_word_g]+delta
	c_s_t[(g_idx,e_idx,len(g),len(e))]=c_s_t[(g_idx,e_idx,len(g),len(e))]+delta
	c_t[(e_idx,len(g),len(e))]=c_t[(e_idx,len(g),len(e))]+delta
  for target_word_e in t_e_g.keys():
    for source_word_g in t_e_g[target_word_e].keys():
      t_e_g[target_word_e][source_word_g]=c_e_g[target_word_e][source_word_g]/c_g[source_word_g]
  for (j,i,l,m) in c_s_t.keys():
    q[(j,i,l,m)]=c_s_t[(j,i,l,m)]/c_t[(i,l,m)]
  
for(g,e) in test_text:
  for (e_idx, target_word_e) in enumerate(e):
    best_idx=0
    score=0
    for (g_idx, source_word_g) in enumerate(g):
      if q[(g_idx,e_idx,len(g),len(e))]*t_e_g[target_word_e][source_word_g]>score:
	best_idx=g_idx
	score=q[(g_idx,e_idx,len(g),len(e))]*t_e_g[target_word_e][source_word_g]
    sys.stdout.write("%i-%i " % (best_idx,e_idx))
  sys.stdout.write("\n") 
"""

      



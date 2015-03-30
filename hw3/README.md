In mydecode.py I tried switch two adjacent translations based on the lm score. In each hypothesis object, I recored the starting and ending point of previous and current hypothesis. The final result is got by backtracking. This implementation gives some improvement but still far away from baseline.

(Thanks for helpful discussion with Chucheng!) In mydecode3.py I tried to insert current translation into  previous translations as well as append it after previous translations. Here I store all translation up to current point in hypothesis object so the final result can just be read out from best hypothesis.



There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model


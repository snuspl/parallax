top_N = 250

import codecs
import os
import random

vocab = []
with codecs.open('/cmsdata/ssd1/cmslab/lm1b/1b_word_vocab.txt', "r", "utf-8") as f:
    for line in f:
        word, count = line.strip().split()
        vocab.append(word)
        if len(vocab) >= top_N:
            break
try:
   os.makedirs('lm1b_synthetic_top%d_words' % top_N)
except:
   pass

with codecs.open('lm1b_synthetic_top%d_words/news.en-00001-of-00100' % top_N, "w", "utf-8") as f:
    epoch = 128*410
    total_size = 0
    while total_size < epoch:
        size = random.randrange(15, 50)
        total_size += size
        words = [vocab[random.randrange(0, top_N)] for i in range(size)]
        f.write(' '.join(words))
        if i < epoch - 1:
            f.write('\n')

from shutil import copyfile
src = 'lm1b_synthetic_top%d_words/news.en-00001-of-00100' % top_N
for i in range(2, 48):
    dst = 'lm1b_synthetic_top%d_words/news.en-%05d-of-00100' % (top_N, i)
    copyfile(src, dst)

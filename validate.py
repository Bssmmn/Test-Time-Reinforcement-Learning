#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import Counter

with open("text8", "r") as f:
    text = f.read()
tokens = text.split()
freq = Counter(tokens)
vocab_size = 10000
vocab = [word for word, count in freq.most_common(vocab_size)]
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}

W_in = np.load("W_in.npy")
print(f"Embeddings loaded  shape: {W_in.shape}")


# In[4]:


def most_similar(word, top_n=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    idx = word2idx[word]
    vec = W_in[idx]
    norms = np.linalg.norm(W_in, axis=1)
    sims = np.dot(W_in, vec) / (norms * np.linalg.norm(vec) + 1e-8)
    top_ids = np.argsort(sims)[::-1][1:top_n+1]
    print(f"\nMost similar to '{word}':")
    for i in top_ids:
        print(f"  {idx2word[i]:20s}  {sims[i]:.4f}")

most_similar("king")
most_similar("france")
most_similar("computer")
most_similar("woman")


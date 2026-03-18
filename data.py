#!/usr/bin/env python
# coding: utf-8

# In[2]:


import urllib.request
import zipfile

url = "http://mattmahoney.net/dc/text8.zip"
urllib.request.urlretrieve(url, "text8.zip")

with zipfile.ZipFile("text8.zip") as f:
    f.extract("text8", ".")

print("Done — file 'text8' is ready")


# In[3]:


# Step 1 — load the raw text
with open("text8", "r") as f:
    text = f.read()

# Step 2 — tokenize (it's already clean, just split)
tokens = text.split()
print(f"Total tokens: {len(tokens)}")  # ~17 million words

# Step 3 — check it looks right
print(tokens[:20])
# ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', ...]


# In[5]:


from collections import Counter

# count every word
freq = Counter(tokens)
print(f"Unique words: {len(freq)}")  # ~250k

# keep top 10,000 most frequent words
vocab_size = 10000
vocab = [word for word, count in freq.most_common(vocab_size)]

# word <-> index mappings
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}

# convert tokens to integers, skip unknown words
token_ids = [word2idx[t] for t in tokens if t in word2idx]

print(f"Vocab size: {len(word2idx)}")
print(f"Tokens after filtering: {len(token_ids)}")
print(f"Sample: {token_ids[:10]}")


# In[8]:


import numpy as np

def generate_skipgram_pairs(token_ids, window_size=5):
    pairs = []
    for i, center in enumerate(token_ids):
        # pick a random window size 1..window_size (as in original paper)
        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(len(token_ids), i + w + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs

# test on a small slice first before running on full dataset
sample_pairs = generate_skipgram_pairs(token_ids[:1000], window_size=5)
print(f"Pairs from first 1000 tokens: {len(sample_pairs)}")
print(f"Example pairs: {sample_pairs[:5]}")



# In[14]:


from collections import Counter

# build vocabulary first
freq = Counter(tokens)
vocab_size = 10000
vocab = [word for word, count in freq.most_common(vocab_size)]
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
token_ids = [word2idx[t] for t in tokens if t in word2idx]

print(f"Vocab size: {len(word2idx)}")
print(f"Tokens after filtering: {len(token_ids)}")

# build negative sampling table (freq ^ 0.75 as in original paper)
freq_array = np.array([freq[idx2word[i]] for i in range(vocab_size)], dtype=np.float32)
freq_array = freq_array ** 0.75
freq_array = freq_array / freq_array.sum()

def negative_sample(k=5):
    return np.random.choice(vocab_size, size=k, p=freq_array)

# test it
samples = negative_sample(5)
print("Negative sample indices:", samples)
print("Negative sample words:", [idx2word[i] for i in samples])


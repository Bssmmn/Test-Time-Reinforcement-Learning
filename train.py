#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# reset matrices fresh
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
token_ids = [word2idx[t] for t in tokens if t in word2idx]

freq_array = np.array([freq[idx2word[i]] for i in range(vocab_size)], dtype=np.float32)
freq_array = freq_array ** 0.75
freq_array = freq_array / freq_array.sum()

def negative_sample(k=5):
    return np.random.choice(vocab_size, size=k, p=freq_array)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

embed_dim = 100
np.random.seed(42)
W_in  = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
W_out = np.zeros((vocab_size, embed_dim))

print("Reset done")


# In[ ]:


train_tokens = np.array(token_ids[:200_000])
window_size = 2
n_negative = 5
epochs = 3

# correct: decay based on actual pair count not token count
initial_lr = 0.025
step = 0
# estimate total pairs
total_steps = len(train_tokens) * window_size * 2 * epochs

for epoch in range(epochs):
    total_loss = 0
    count = 0

    for i in range(len(train_tokens)):
        center_id = train_tokens[i]

        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(len(train_tokens), i + w + 1)
        context_ids = np.concatenate([train_tokens[start:i], train_tokens[i+1:end]])

        for context_id in context_ids:
            # correct learning rate decay
            lr = max(initial_lr * 0.0001, initial_lr * (1 - step / total_steps))
            
            neg_ids = negative_sample(n_negative)

            v_c = W_in[center_id]
            v_o = W_out[context_id]
            v_n = W_out[neg_ids]

            pos_sig = sigmoid(np.dot(v_c, v_o))
            neg_sig = sigmoid(np.dot(v_n, v_c))

            loss = -np.log(pos_sig + 1e-7) - np.sum(np.log(1 - neg_sig + 1e-7))
            total_loss += loss
            count += 1

            pos_err = pos_sig - 1
            neg_err = neg_sig

            grad_v_c = pos_err * v_o + neg_err @ v_n
            W_out[context_id] -= lr * pos_err * v_c
            W_out[neg_ids]    -= lr * np.outer(neg_err, v_c)
            W_in[center_id]   -= lr * grad_v_c

            step += 1

    avg_loss = total_loss / count
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Final LR: {lr:.6f}")

print("Training done ")
np.save("W_in.npy", W_in)
print("Embeddings saved ")


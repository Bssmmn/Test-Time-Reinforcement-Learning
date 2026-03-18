#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# hyperparameters
vocab_size = 10000
embed_dim = 100
learning_rate = 0.025

# two matrices — this is the entire "model"
# W_in  = embedding for center words
# W_out = embedding for context words
np.random.seed(42)
W_in  = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
W_out = np.zeros((vocab_size, embed_dim))

print(f"W_in shape:  {W_in.shape}")   # (10000, 100)
print(f"W_out shape: {W_out.shape}")  # (10000, 100)


# In[3]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(center_id, context_id, neg_ids):
    # look up vectors
    v_c = W_in[center_id]                # shape (100,)
    v_o = W_out[context_id]              # shape (100,)
    v_n = W_out[neg_ids]                 # shape (5, 100)

    # dot products
    pos_score = np.dot(v_c, v_o)         # one number
    neg_scores = np.dot(v_n, v_c)        # 5 numbers

    # sigmoid
    pos_sig = sigmoid(pos_score)         # should go → 1
    neg_sig = sigmoid(neg_scores)        # should go → 0

    # loss (we want to minimize this)
    loss = -np.log(pos_sig) - np.sum(np.log(1 - neg_sig))

    return loss, v_c, v_o, v_n, pos_sig, neg_sig


# In[4]:


def backward(center_id, context_id, neg_ids, v_c, v_o, v_n, pos_sig, neg_sig):
    # how wrong were we?
    pos_error = pos_sig - 1              # want pos_sig = 1, so error = sig - 1
    neg_error = neg_sig                  # want neg_sig = 0, so error = sig - 0

    # gradients
    grad_v_c = pos_error * v_o + np.dot(neg_error, v_n)
    grad_v_o = pos_error * v_c
    grad_v_n = np.outer(neg_error, v_c)

    # update matrices
    W_in[center_id]  -= learning_rate * grad_v_c
    W_out[context_id] -= learning_rate * grad_v_o
    W_out[neg_ids]   -= learning_rate * grad_v_n


# In[5]:


# get one real pair from data.ipynb to test
center_id  = 5233   # from your sample_pairs earlier
context_id = 3080
neg_ids    = np.array([0, 1, 2, 3, 4])  # fake negatives for now

loss, v_c, v_o, v_n, pos_sig, neg_sig = forward(center_id, context_id, neg_ids)
print(f"Loss: {loss:.4f}")
print(f"Positive score (want → 1): {pos_sig:.4f}")
print(f"Negative scores (want → 0): {neg_sig}")

backward(center_id, context_id, neg_ids, v_c, v_o, v_n, pos_sig, neg_sig)
print("Backward pass done — no errors means gradients work ✅")


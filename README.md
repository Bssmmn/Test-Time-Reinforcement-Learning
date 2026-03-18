# Word2Vec from Scratch

Implementation of Word2Vec (Skip-Gram with Negative Sampling) using only NumPy.
No PyTorch, no TensorFlow, nothing — just matrix math.

## Why I built it this way

I wanted to actually understand what's happening inside word embeddings,
not just call model.fit(). Writing the forward pass, loss, and gradients
by hand forces you to understand every step.

## How it works

The model maintains two matrices W_in and W_out (10,000 × 100).
For each word in the text, it looks at neighboring words in a window
and tries to maximize the dot product between words that appear together,
while minimizing it for randomly sampled "noise" words.

Do this 14 million times and the vectors start to mean something.

## Dataset

text8 — a cleaned Wikipedia dump, ~17 million words.
Standard benchmark for word embedding models.

## How to run
```
1. data.ipynb   — downloads text8, builds vocabulary
2. model.ipynb  — defines forward pass and gradients
3. train.ipynb  — runs training loop
4. validate.ipynb — checks nearest neighbors
```

## Results

Training loss: 2.92 → 2.37 → 2.28 across 3 epochs
```
most_similar("france") → foreign, cambridge, politics, books
most_similar("computer") → processing, methodology
```

Trained on 200k tokens. Better results with full dataset but takes hours on CPU.

## What I'd improve with more time

- Train on the full 17M tokens
- Add proper analogy evaluation (king - man + woman = queen)
- Vectorize the inner loop with NumPy to speed up training
- Try hierarchical softmax as alternative to negative sampling
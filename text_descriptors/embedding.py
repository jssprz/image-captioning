import nltk
import numpy as np
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=300):
        super(WordEmbedding, self).__init__()

        self.embeds = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.out_size = embedding_dim

    def forward(self, idx_texts):
        return self.embeds(idx_texts)


class MeanEmbedding:
    def __init__(self, embeddings, embedding_dim):
        self.embeddings = embeddings
        self.out_size = embedding_dim

    def transform(self, texts):
        return np.array([np.mean(np.array([self.embeddings[w] for w in nltk.tokenize.word_tokenize(t) if w in self.embeddings]), axis=0) for t in texts])

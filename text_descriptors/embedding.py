import nltk
import numpy as np
import torch.nn as nn


class MyWordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=300):
        super(MyWordEmbedding, self).__init__()
        self.embeds = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.out_size = embedding_dim

    def forward(self, idx_texts):
        return self.embeds(idx_texts)


class WordEmbedding:
    def __init__(self, embeddings, embedding_dim, lowercase=True):
        self.embeddings = embeddings
        self.out_size = embedding_dim
        self.lowercase = lowercase

    def transform(self, texts, mode='mean'):
        if mode == 'mean':
            return np.array([np.mean(
                [self.embeddings[w] for w in nltk.tokenize.word_tokenize(t.lower() if self.lowercase else t) if
                 w in self.embeddings], axis=0) for t in texts])
        else:
            result = []
            for t in texts:
                tokens = nltk.tokenize.word_tokenize(t.lower() if self.lowercase else t)
                while len(tokens) < 20:
                    tokens.append('fin')
                if len(tokens) > 20:
                    tokens = tokens[:20]
                result.append([self.embeddings['algo'] if w not in self.embeddings else self.embeddings[w] for w in tokens])
            return np.array(result)

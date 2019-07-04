import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer


class WordEmbedding(nn.Module):
    def __init__(self, texts, lowecase, ngrams_range, max_df, min_df, embedding_dim=300, use_pretrained=False):
        super(WordEmbedding, self).__init__()

        self.cv = CountVectorizer(lowercase=lowecase, ngrams_range=ngrams_range, max_df=max_df,
                                  min_df=min_df, binary=True)
        self.cv.fit_transform(texts)
        num_embeddings = len(self.cv.vocabulary_)

        self.embeds = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.out_size = embedding_dim

    def word_to_idx(self, texts):
        return [[self.cv.vocabulary_[w] for w in t.strip().split(' ')] for t in texts]

    def forward(self, idx_texts):
        return self.embeds(idx_texts)

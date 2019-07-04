import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim=300, use_pretrained=False):
        super(WordEmbedding, self).__init__()

        self.embeds = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

        self.out_size = embedding_dim

    def forward(self, idx_texts):
        return self.embeds(idx_texts)

from sklearn.decomposition import TruncatedSVD
from .bow import TextDescriptor


class LSADescriptor:
    def __init__(self, type, texts, lowecase, ngrams_range, max_df, min_df, n_components=10):
        self.text_descriptor = TextDescriptor(type, texts, lowecase, ngrams_range, max_df, min_df)
        self.lsa = TruncatedSVD(n_components=n_components, algorithm='arpack')
        self.out_size = self.text_descriptor.out_size

    def transform(self, texts):
        a = self.text_descriptor.transform(texts)
        return self.lsa.transform(a)

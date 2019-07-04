from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


class TextDescriptor:
    def __init__(self, type, texts, lowecase, ngrams_range, max_df, min_df):
        if type == 'bow':
            self.descriptor = CountVectorizer(lowercase=lowecase, ngrams_range=ngrams_range, max_df=max_df,
                                              min_df=min_df, binary=True)
        elif type == 'tf-idf':
            self.descriptor = TfidfVectorizer(lowercase=lowecase, ngram_range=ngrams_range, max_df=max_df,
                                              min_df=min_df, norm=None)
        else:
            raise '{} is an unknown type of descriptor'.format(type)

        self.descriptor.fit(texts)

        self.out_size = len(self.descriptor.vocabulary_)

    def transform(self, texts):
        return self.descriptor.transform(texts)

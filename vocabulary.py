#!/usr/bin/env python
"""Defines the Vocabulary class
"""

import nltk

nltk.download('punkt')
from collections import Counter


class Vocabulary(object):
    def __init__(self, lowercase=True, max_df=1, min_df=0):
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.nwords = 0
        self.lowercase = lowercase
        self.max_df = max_df
        self.min_df = min_df

    @classmethod
    def from_words(cls, words):
        result = cls()
        for w in words:
            result.add_word(w)
        return result

    def add_sentences(self, sentences):
        """
        Get a glossary based on the text of the annotation.
        Words with frequencies lower than threshold will be omitted
        """
        counter = Counter()
        ncaptions = len(sentences)
        for i, caption in enumerate(sentences):
            # Segmenting words directly by space
            # tokens = caption.lower().split(' ')
            # Use nltk for word segmentation
            tokens = nltk.tokenize.word_tokenize(caption.lower() if self.lowercase else caption)
            counter.update(tokens)
            if i % 10000 == 0:
                print('[{}/{}] tokenized the captions.'.format(i, ncaptions))

        for i, (w, c) in enumerate(counter.items(), start=self.nwords):
            df = c / ncaptions
            if self.min_df <= df <= self.max_df:  # Skip some low and high frequency words
                self.word2idx[w] = i
                self.idx2word[i] = w
                self.word2count[w] = c
                self.nwords += 1

    def add_word(self, w):
        """
        Add new words to the glossary
        """
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.word2count[w] = 1
            self.idx2word[self.nwords] = w
            self.nwords += 1
        else:
            self.word2count[w] += 1

    def add_words(self, words):
        for w in words:
            self.add_word(w)

    def add_sentence(self, s):
        self.add_words(s.split(' '))

    def remove_words(self, words):
        removed_words = []
        for word in words:
            if word in self.word2idx:
                idx = self.word2idx[word]
                del self.word2idx[word]
                del self.idx2word[idx]
                del self.word2count[word]
                self.nwords -= 1
                removed_words.append(word)
        return removed_words

    def __call__(self, texts):
        """
        Returns the id corresponding to the word
        """
        return [[self.word2idx['<unk>'] if w not in self.word2idx else self.word2idx[w] for w in t] for t in texts]

    def __len__(self):
        """
        Get the number of words in the vocabulary
        """
        return self.nwords

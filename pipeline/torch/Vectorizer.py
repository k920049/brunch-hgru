import string
from collections import Counter

import numpy as np

from pipeline.torch.Vocabulary import Vocabulary

class Vectorizer(object):

    def __init__(self,
                 review_vocab,
                 rating_vocab):

        self.review_vocab = review_vocab
        self.rating_vocab

    def vectorize(self, review):

        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):

        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):

        return {
            'review_vocab': self.review_vocab.to_serializable(),
            'rating_vocab': self.rating_vocab.to_serializable()
        }
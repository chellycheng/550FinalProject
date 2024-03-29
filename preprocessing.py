import numpy as np
import torch
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

def generate_pair(corpus):
    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)

    window_size = 2
    idx_pairs = []
    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs)  # it will be useful to have this as numpy array
    return idx_pairs

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lable, corpus):
        if lable == 'train':
            self.word_pair_lists = generate_pair(corpus)

        elif lable == 'test':
            self.word_pair_lists = generate_pair(corpus)
            self.labels = None

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = -1
        image = self.images[index]
        image = Image.fromarray(image.reshape(128, 128))
        if self.labels is not None:
            label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
            image -= image.mean()
            image /= image.std()
            image /= 255
        return image, label

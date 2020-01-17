import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
# from PIL import Image
from build_vocab import Vocabulary
import torch.nn as nn

import pandas as pd

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, dictionary, emb_model):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
        """
        # Read the dataset
        self.data = pd.read_csv(json, header=0,encoding = 'unicode_escape',error_bad_lines=False)
        self.ids = list(range(len(self.data)))
        self.emb_model = emb_model
        self.vocab = self.emb_model.wv.vocab
        # All the keywords present
        # dictionary = pd.read_csv(dictionary, header=0,encoding = 'unicode_escape',error_bad_lines=False)
        dictionary =pd.read_csv(dictionary, header=0,error_bad_lines=False)
        self.dictionary = list(dictionary['keys'])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        data = self.data
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = data.iloc[ann_id]['val']

        array = torch.zeros((256))
        keywords = self.data.iloc[index]['tk'].split()
        #
        for val in keywords:
            array = torch.add(array, torch.from_numpy(self.emb_model.wv[val]))
            # print("in data_loader", array)
        array = torch.div(array, len(keywords))

        # Convert caption (string) to word ids.
        # print("caption before tokenize", caption)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # print("tokens", tokens)
        caption = []
        caption.append(vocab['<start>'].index)
        caption.extend([vocab[token].index if token in vocab else vocab['<unk>'].index for token in tokens])
        caption.append(vocab['<end>'].index)
        target = torch.Tensor(caption)
        # print("target in __getitem__", target)
        return array, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - array: torch tensor of shape (len(dictionary)).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        array: torch tensor of shape (batch_size, len(dictionary)).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    # print("Data", data)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # print("In collate_fn")
    array, captions = zip(*data)
    # print("captions", captions)

    # Merge arrays (from tuple of 3D tensor to 4D tensor).
    # print("After unzip", type(array))
    array = torch.stack(array, 0)
    # print("Final", array)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # print("targest", targets)
    # embed = nn.Embedding(8594, 256)
    # embeddings = embed(targets)
    # print("shape ",embeddings)
    return array, targets, lengths

def get_loader(root, json, dictionary, emb_model, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                        json=json,
                        dictionary=dictionary,
                        emb_model = emb_model)

    # Data loader for COCO dataset
    # This will return (array, captions, lengths) for each iteration.
    # array: a tensor of shape (batch_size, len(dictionary)).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

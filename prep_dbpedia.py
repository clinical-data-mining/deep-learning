import logging
import pickle
import torch
import torchtext
from torchtext.datasets import text_classification
from torchtext.vocab import build_vocab_from_iterator
import os
NGRAMS = 2
PATH_OUTPUT = 'output/'


# parse and save dataset
def prepare_dataset(ngrams=1):
    extracted_dir = '/home/rajannaa/pytorch-demo/data/dbpedia_csv/';

    vocab = None
    train_csv_path = os.path.join(extracted_dir, 'train.csv')
    test_csv_path = os.path.join(extracted_dir, 'test.csv')

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(text_classification._csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), False)
    logging.info('Creating testing data')
    test_data, test_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), False)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")

    train_dataset = text_classification.TextClassificationDataset(vocab, train_data, train_labels)
    test_dataset = text_classification.TextClassificationDataset(vocab, test_data, test_labels)
 
    pickle.dump(train_dataset, open(os.path.join(PATH_OUTPUT, "train_dataset"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_dataset, open(os.path.join(PATH_OUTPUT, "test_dataset"), 'wb'), pickle.HIGHEST_PROTOCOL)


   
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

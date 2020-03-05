import torch
import torchtext
from torchtext.datasets import text_classification
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from model import TextSentiment

train_dataset = pickle.load(open(os.path.join("./data/dbpedia_csv", "train_dataset"), 'rb'))
test_dataset = pickle.load(open(os.path.join("./data/dbpedia_csv", "test_dataset"), 'rb'))

BATCH_SIZE = 16
N_EPOCHS = 5
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASS = len(train_dataset.get_labels())
HARDWARE = 'GPU'

min_valid_loss = float('inf')
device = torch.device("cuda" if HARDWARE == 'GPU'  else "cpu")
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

print('2. Train on {0} for {1} epochs'.format(HARDWARE, N_EPOCHS))
print('Loss function is', criterion)
print('Model is\n', model)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()
    
    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

def main():
    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print('Save trained model to run on cpu')
    torch.save(model.to('cpu').state_dict(), 'model_ep.pth')

if __name__ == "__main__" :
    main()


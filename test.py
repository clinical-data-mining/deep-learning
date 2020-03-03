import torch
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import pickle
import os
import torch.nn as nn
import torch.nn.functional as F
from model import TextSentiment

def predict(text, model, vocab, ngrams, label_map):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
	
        top5_prob, top5_label = torch.topk(output, 5)
        for i in range(5):
            print(label_map[top5_label.data[0][i].item()+1], top5_prob.data[0][i].item())
        return output.argmax(1).item() + 1

def main():
    train_dataset = pickle.load(open(os.path.join("./data", "train_dataset"), 'rb'))
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUM_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)
    label = {1 : "Company", 2 : "EducationalInstitution", 3 : "Artist", 4 : "Athlete", 5 : "OfficeHolder", 6 : "MeanOfTransportation", 7 : "Building", 8 : "NaturalPlace", 9 : "Village", 10 : "Animal", 11 : "Plant", 12 : "Album", 13 : "Film", 14 : "WrittenWork"}

    model.load_state_dict(torch.load("./model_ep.pth"))
    model.eval()
    model = model.to("cpu")
    print(model)
    ex_text_str = "Memorial Sloan Kettering Cancer Center (MSK or MSKCC) is a cancer treatment and research institution in New York City, founded in 1884 as the New York Cancer Hospital. MSKCC is the largest and oldest private cancer center in the world, and is one of 70 National Cancer Institute–designated Comprehensive Cancer Centers. Its main campus is located at 1275 York Avenue, between 67th and 68th Streets, in Manhattan."

    vocab = train_dataset.get_vocab()
    print("This is a %s" %label[predict(ex_text_str, model, vocab, 2, label)])

if __name__ == "__main__" :
    main()


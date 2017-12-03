import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vectorizer import build_features_lower, vectorize_lower, defaultdict
import numpy as np
import random
import argparse
import pickle
import os.path
from datetime import datetime

torch.manual_seed(1)

def save_model(model, word_id, tag_id, cells, file_name):
    with open(file_name, "w") as f:
        pickle.dump([model, word_id, tag_id, cells], f, pickle.HIGHEST_PROTOCOL)

def load_model(file_name):
    if os.path.isfile(file_name):
        content = pickle.load(open(file_name))
        return content[0], content[1], content[2], content[3]
    else:
        print "Model Not Found"

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, embeds):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embeds))

        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(1)
        self.E = autograd.Variable(torch.zeros(vocab_size, hidden_dim))
        self.C = autograd.Variable(torch.zeros(vocab_size, hidden_dim))

    def init_hidden(self, size):
        return (autograd.Variable(torch.zeros(size, self.hidden_dim)),
                autograd.Variable(torch.zeros(size, self.hidden_dim)))

    def forward(self, sentences):#all path segments in this batch have same size
        embeds = self.word_embeddings(sentences)
        lstm_out = autograd.Variable(torch.zeros(len(sentences[0]), len(sentences), self.hidden_dim))
        cells = autograd.Variable(torch.zeros(len(sentences[0]), len(sentences), self.hidden_dim))
        # c_pt = autograd.Variable(torch.zeros(len(sentences), self.hidden_dim))
        for t in range(len(sentences[0])):
            h_t, c_t = self.lstm_cell(embeds[:,t], self.hidden)
            # c_delta = c_t - c_pt
            # for i in range(len(sentences)):
            #     idx = sentences[i][t]
            #     self.E[idx] += c_delta[i]
            #     self.C[idx] += 1
            # c_pt = c_t
            self.hidden = (h_t, c_t)
            lstm_out[t] = h_t
            cells[t] = c_t
        tag_space = self.hidden2tag(lstm_out.view(len(sentences[0])*len(sentences), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores, cells

def get_pretrained_embedings(embeds_file, word_id, EMBEDDING_DIM):
    valid = []
    embeds = np.zeros((len(word_id), EMBEDDING_DIM))
    for line in open(embeds_file):
        s = line.split(" ")
        if s[0] in word_id:
            embed = map(float, s[1:])
            embeds[word_id[s[0]]] = np.array(embed)
            valid.append(s[0])
    for s in word_id:
        if s not in valid:
            embeds[word_id[s]] = embeds[0] = np.mean(embeds, axis=0)
    return embeds

def train(EMBEDDING_DIM, HIDDEN_DIM, word_to_ix, tag_to_ix, training_data, embeds):
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), embeds)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    for epoch in range(1):  
        print "start epoch "+str(epoch)+" "+str(datetime.now())
        keys = training_data.keys()
        # random.shuffle(keys)
        for i in keys:
            train_mini = training_data[i]
            # random.shuffle(train_mini)
            model.zero_grad()
            model.hidden = model.init_hidden(len(train_mini))
            sentences = zip(*train_mini)[0]
            tags = zip(*train_mini)[1]

            sentences_in = autograd.Variable(torch.LongTensor(sentences))
            targets = autograd.Variable(torch.LongTensor(tags))

            tag_scores, cells = model(sentences_in)
            print tag_scores
            loss = loss_function(tag_scores, targets.view(-1))
            print loss
            
            loss.backward()
            optimizer.step()
    return model, cells

def predict(testing_data, model, target_size):
    predictions = list()
    for sentence, tags in testing_data:
        sentence_in = autograd.Variable(torch.LongTensor([sentence]))
        targets = np.array(tags)
        model.hidden = model.init_hidden(1)
        prediction = torch.max(model(sentence_in)[0].view(-1, target_size).data, 1)[1].numpy().tolist()
        predictions.append(prediction)
    return predictions

def test(testing_data, model, target_size):
    total = 0.0
    correct = 0.0
    total_un = 0.0
    correct_un = 0.0
    predictions = predict(testing_data, model, target_size)
    for i, s_t in enumerate(testing_data):
        sentence = s_t[0]
        tags = s_t[1]
        for j, tag in enumerate(tags):
            total += 1.0
            if tag == predictions[i][j]:
                correct += 1.0
            if tag == 0:
                total_un += 1.0
                if tag == predictions[i][j]:
                    correct_un += 1.0
    return correct/total, correct_un/total_un

if __name__ == "__main__":
    # set all arguments we need
    parser = argparse.ArgumentParser(description='Parameters for Markov model (MM) POS tagger')
    parser.add_argument("--load", dest="load", type=str, default="")
    parser.add_argument("--save", dest="file_name", type=str, default="")
    args = parser.parse_args()
    
    file_name = args.file_name
    load = args.load

    if load == "":
        word_id, id_word, tag_id, id_tag = build_features_lower("PTBSmall", "train.tagged")
        embeds = get_pretrained_embedings("glove.6B.200d.txt", word_id, 200)
        training_v = vectorize_lower("PTBSmall", "train.tagged", word_id, tag_id)
        training_data = defaultdict(list)
        for v in training_v:
            training_data[len(v[0])].append(v)
        model, cells = train(200, 50, word_id, tag_id, training_data, embeds)
        if file_name != "":
            save_model(model, word_id, tag_id, cells, file_name)
    else:
        model, word_id, tag_id, cells = load_model(load)
    dev = vectorize_lower("PTBSmall", "test.tagged", word_id, tag_id)
    print test(dev, model, len(tag_id))

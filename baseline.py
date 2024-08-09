from torch import nn
import torch
import numpy as np
import pandas as pd
import pickle, time
import re, os, string, typing, gc, json
import torch.nn.functional as F
import spacy
from sklearn.model_selection import train_test_split
from collections import Counter
nlp = spacy.load('en')
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from preprocess import *


train_data = load_json('/kaggle/input/squad-data/data/squad_train.json')
valid_data = load_json('/kaggle/input/squad-data/data/squad_dev.json')

train_list = parse_data(train_data)
valid_list = parse_data(valid_data)

train_df = pd.DataFrame(train_list)
valid_df = pd.DataFrame(valid_list)

def preprocess_df(df):
    
    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df.answer = df.answer.apply(to_lower)

preprocess_df(train_df)
preprocess_df(valid_df)

vocab_text = gather_text_for_vocab([train_df, valid_df])

word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)

train_df['context_ids'] = train_df.context.apply(context_to_ids, word2idx=word2idx)
valid_df['context_ids'] = valid_df.context.apply(context_to_ids, word2idx=word2idx)
train_df['question_ids'] = train_df.question.apply(question_to_ids, word2idx=word2idx)
valid_df['question_ids'] = valid_df.question.apply(question_to_ids, word2idx=word2idx)

train_err = get_error_indices(train_df, idx2word)
valid_err = get_error_indices(valid_df, idx2word)

train_df.drop(train_err, inplace=True)
valid_df.drop(valid_err, inplace=True)

train_label_idx = train_df.apply(index_answer, axis=1, idx2word=idx2word)
valid_label_idx = valid_df.apply(index_answer, axis=1, idx2word=idx2word)

train_df['label_idx'] = train_label_idx
valid_df['label_idx'] = valid_label_idx


train_df.to_pickle('basetrain.pkl')
valid_df.to_pickle('basevalid.pkl')

with open('basew2id.pickle','wb') as handle:
    pickle.dump(word2idx, handle)

train_df = pd.read_pickle('iNLP/data/preproc/basetrain.pkl')
valid_df = pd.read_pickle('iNLP/data/preproc/basevalid.pkl')

with open('iNLP/data/preproc/basew2id.pickle','rb') as handle:
    word2idx = pickle.load(handle)

idx2word = {v:k for k,v in word2idx.items()}


class SquadDataset:  
    def __init__(self, data, batch_size):
        
        self.batch_size = batch_size
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data
    
    def get_span(self, text):
        
        text = nlp(text, disable=['parser','tagger','ner'])
        span = [(w.idx, w.idx+len(w.text)) for w in text]

        return span

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):        
        for batch in self.data:
                            
            spans = []
            context_text = []
            answer_text = []
            
            max_context_len = max([len(ctx) for ctx in batch.context_ids])
            padded_context = torch.LongTensor(len(batch), max_context_len).fill_(1)
            
            for ctx in batch.context:
                context_text.append(ctx)
                spans.append(self.get_span(ctx))
            
            for ans in batch.answer:
                answer_text.append(ans)
                
            for i, ctx in enumerate(batch.context_ids):
                padded_context[i, :len(ctx)] = torch.LongTensor(ctx)
            
            max_question_len = max([len(ques) for ques in batch.question_ids])
            padded_question = torch.LongTensor(len(batch), max_question_len).fill_(1)
            
            for i, ques in enumerate(batch.question_ids):
                padded_question[i,: len(ques)] = torch.LongTensor(ques)
                
            
            label = torch.LongTensor(list(batch.label_idx))
            
            ids = list(batch.id)  
            
            yield (padded_context, padded_question, label, context_text, answer_text, ids)


train_dataset = SquadDataset(train_df, 16)

valid_dataset = SquadDataset(valid_df, 16)

def get_glove_dict():
    glove_dict = {}
    with open("./glove-emd/glove.6B.300d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_dict[word] = vector
            
    f.close()
    
    return glove_dict

glove_dict = get_glove_dict()

def create_weights_matrix(glove_dict):
    weights_matrix = np.zeros((len(word_vocab), 300))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            pass
        
    return weights_matrix, words_found

weights_matrix, words_found = create_weights_matrix(glove_dict)

np.save('baseglove.npy', weights_matrix)

class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                
    def forward(self, x):
        outputs, _ = self.lstm(x)
        
        return outputs
    
class Baseline(nn.Module):
    
    def __init__(self, emb_dim, ctx_hidden_dim, device):
        super().__init__()
        
        self.device = device
        
        self.word_embedding = self.get_glove_embedding()
        
        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim, ctx_hidden_dim)
                
        self.similarity_weight = nn.Linear(emb_dim*3, 1, bias=False)
        
        self.modeling_lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)
        
        self.output_start = nn.Linear(emb_dim*2, 1, bias=False)
        
        self.output_end = nn.Linear(emb_dim*2, 1, bias=False)        
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('iNLP/bidafglove_tv.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=True)

        return embedding
        
    def forward(self, ctx, ques):        
        ctx_len = ctx.shape[1]
        
        ques_len = ques.shape[1]
        
        ctx_word_embed = self.word_embedding(ctx)
        
        ques_word_embed = self.word_embedding(ques)
        
        ctx_contextual_emb = self.contextual_embedding(ctx_word_embed)
        
        ques_contextual_emb = self.contextual_embedding(ques_word_embed)
        
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1,1,ques_len,1)

        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1,ctx_len,1,1)

        elementwise_prod = torch.mul(ctx_, ques_)

        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)

        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        
        a = F.softmax(similarity_matrix, dim=-1)
        
        c2q = torch.bmm(a, ques_contextual_emb)
        
        G = torch.mul(ctx_contextual_emb,c2q)
        
        M, _ = self.modeling_lstm(G)
        
        M2, _ = self.modeling_lstm(M)
        
        p1 = self.output_start(torch.cat([G,M], dim=2))
        
        p1 = p1.squeeze()
        
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        
        return p1, p2

EMB_DIM = 100
HIDDEN_DIM = 100
device = torch.device('cuda')

model = Baseline(EMB_DIM,
              HIDDEN_DIM, 
              device).to(device)

optimizer = optim.Adadelta(model.parameters())

def train(model, train_dataset):
    print("Starting training ........")
   
    train_loss = 0.
    model.train()
    for batch in tqdm(train_dataset):
        
        optimizer.zero_grad()

        context, question, label, ctx_text, ans, ids = batch

        context, question, label = context.to(device), question.to(device), label.to(device)


        preds = model(context, question)

        start_pred, end_pred = preds

        s_idx, e_idx = label[:,0], label[:,1]

        loss = F.cross_entropy(start_pred, s_idx) + F.cross_entropy(end_pred, e_idx)

        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()

    return train_loss/len(train_dataset)


def valid(model, valid_dataset):
    
    print("Starting validation .........")
   
    valid_loss = 0.

    f1, em = 0., 0.
    
    model.eval()
        
   
    predictions = {}
    
    for batch in tqdm(valid_dataset):

        context, question, label, ctx, answers, ids = batch

        context, question, label = context.to(device), question.to(device), label.to(device)
        
        with torch.no_grad():
            
            s_idx, e_idx = label[:,0], label[:,1]

            preds = model(context, question)

            p1, p2 = preds

            
            loss = F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)

            valid_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
            
           
            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i]+1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[id] = pred
            

    
    em, f1 = evaluate(predictions)
    return valid_loss/len(valid_dataset), em, f1


def evaluate(predictions):
    with open('iNLP/data/squad_dev.json','r',encoding='utf-8') as f:
        dataset = json.load(f)
        
    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue
                
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                
                prediction = predictions[qa['id']]
                
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
        
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


train_losses = []
valid_losses = []
ems = []
f1s = []
epochs = 8
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    start_time = time.time()
    
    train_loss = train(model, train_dataset)
    valid_loss, em, f1 = valid(model, valid_dataset)  
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    ems.append(em)
    f1s.append(f1)

    print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
    print(f"Epoch valid loss: {valid_loss}")
    print(f"Epoch EM: {em}")
    print(f"Epoch F1: {f1}")
    print("====================================================================================")


import matplotlib.pyplot as plt

e = [1,2,3,4,5,6,7,8]

plt.figure()
plt.plot(e, train_losses, marker='o', label='Train Loss')
plt.plot(e, valid_losses, marker='s', label='Valid Loss')
plt.title('Loss vs epochs')
plt.xlabel('Loss')
plt.ylabel('epochs')
plt.legend()
plt.show()
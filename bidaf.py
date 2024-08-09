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
char2idx, char_vocab = build_char_vocab(vocab_text)

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


train_df.to_pickle('bidaftrain.pkl')
valid_df.to_pickle('bidafvalid.pkl')

with open('bidafw2id.pickle','wb') as handle:
    pickle.dump(word2idx, handle)

with open('iNLP/data/bidafc2id.pickle','wb') as handle:
    pickle.dump(char2idx, handle)

train_df = pd.read_pickle('./bidaftrain.pkl')
valid_df = pd.read_pickle('./bidafvalid.pkl')

with open('./bidafw2id.pickle','rb') as handle:
    word2idx = pickle.load(handle)
with open('./bidafc2id.pickle','rb') as handle:
    char2idx = pickle.load(handle)

idx2word = {v:k for k,v in word2idx.items()}


class SquadDataset:
    def __init__(self, data, batch_size):
        
        self.batch_size = batch_size
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data
        
        
    def __len__(self):
        return len(self.data)
    
    def make_char_vector(self, max_sent_len, max_word_len, sentence):
        
        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)
        
        for i, word in enumerate(nlp(sentence, disable=['parser','tagger','ner'])):
            for j, ch in enumerate(word.text):
                char_vec[i][j] = char2idx.get(ch, 0)
        
        return char_vec    
    
    def get_span(self, text):
        
        text = nlp(text, disable=['parser','tagger','ner'])
        span = [(w.idx, w.idx+len(w.text)) for w in text]

        return span

    def __iter__(self):        
        for batch in self.data:
            
            spans = []
            ctx_text = []
            answer_text = []
            
            for ctx in batch.context:
                ctx_text.append(ctx)
                spans.append(self.get_span(ctx))
            
            for ans in batch.answer:
                answer_text.append(ans)
                
            
            max_context_len = max([len(ctx) for ctx in batch.context_ids])
            padded_context = torch.LongTensor(len(batch), max_context_len).fill_(1)
            
            for i, ctx in enumerate(batch.context_ids):
                padded_context[i, :len(ctx)] = torch.LongTensor(ctx)
                
            max_word_ctx = 0
            for context in batch.context:
                for word in nlp(context, disable=['parser','tagger','ner']):
                    if len(word.text) > max_word_ctx:
                        max_word_ctx = len(word.text)
            
            char_ctx = torch.ones(len(batch), max_context_len, max_word_ctx).type(torch.LongTensor)
            for i, context in enumerate(batch.context):
                char_ctx[i] = self.make_char_vector(max_context_len, max_word_ctx, context)
            
            max_question_len = max([len(ques) for ques in batch.question_ids])
            padded_question = torch.LongTensor(len(batch), max_question_len).fill_(1)
            
            for i, ques in enumerate(batch.question_ids):
                padded_question[i, :len(ques)] = torch.LongTensor(ques)
                
            max_word_ques = 0
            for question in batch.question:
                for word in nlp(question, disable=['parser','tagger','ner']):
                    if len(word.text) > max_word_ques:
                        max_word_ques = len(word.text)
            
            char_ques = torch.ones(len(batch), max_question_len, max_word_ques).type(torch.LongTensor)
            for i, question in enumerate(batch.question):
                char_ques[i] = self.make_char_vector(max_question_len, max_word_ques, question)
            
            ids = list(batch.id)  
            label = torch.LongTensor(list(batch.label_idx))
            
            yield (padded_context, padded_question, char_ctx, char_ques, label, ctx_text, answer_text, ids)
            
            

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

np.save('bidafglove.npy', weights_matrix)

#Model

class CharacterEmbeddingLayer(nn.Module):
    
    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):
        
        super().__init__()
        
        self.char_emb_dim = char_emb_dim
        
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)
        
        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)
        
        self.relu = nn.ReLU()
    
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.dropout(self.char_embedding(x))
        x = x.permute(0,1,3,2)
        x = x.view(-1, self.char_emb_dim, x.shape[3])
        x = x.unsqueeze(1)        
        x = self.relu(self.char_convolution(x))
        x = x.squeeze()
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        x = x.view(batch_size, -1, x.shape[-1])
        return x
    
class HighwayNetwork(nn.Module):
    
    def __init__(self, input_dim, num_layers=2):
        
        super().__init__()
        
        self.num_layers = num_layers
        
        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        
        for i in range(self.num_layers):
            
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))
            
            x = gate_value * flow_value + (1-gate_value) * x
        
        return x

class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.highway_net = HighwayNetwork(input_dim)
        
    def forward(self, x):
        highway_out = self.highway_net(x)
        outputs, _ = self.lstm(highway_out)
        return outputs
    
class BiDAF(nn.Module):
    
    def __init__(self, char_vocab_dim, emb_dim, char_emb_dim, num_output_channels, 
                 kernel_size, ctx_hidden_dim, device):
        super().__init__()
        
        self.device = device
        
        self.word_embedding = self.get_glove_embedding()
        
        self.character_embedding = CharacterEmbeddingLayer(char_vocab_dim, char_emb_dim, 
                                                      num_output_channels, kernel_size)
        
        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim*4//3, ctx_hidden_dim)
        
        self.dropout = nn.Dropout()
        
        self.similarity_weight = nn.Linear(emb_dim*2, 1, bias=False)
        
        self.modeling_lstm = nn.LSTM(emb_dim*8//3, emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        
        self.output_start = nn.Linear(emb_dim*14//3, 1, bias=False)
        
        self.output_end = nn.Linear(emb_dim*14//3, 1, bias=False)
        
        self.end_lstm = nn.LSTM(emb_dim*2, emb_dim, bidirectional=True, batch_first=True)
        
    
    def get_glove_embedding(self):
        
        weights_matrix = np.load('baseglove.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=True)

        return embedding
        
    def forward(self, ctx, ques, char_ctx, char_ques):
        ctx_len = ctx.shape[1]
        ques_len = ques.shape[1]
        ctx_word_embed = self.word_embedding(ctx)
        
        ques_word_embed = self.word_embedding(ques)
        
        ctx_char_embed = self.character_embedding(char_ctx)
        
        ques_char_embed = self.character_embedding(char_ques)
        
        ctx_contextual_inp = torch.cat([ctx_word_embed, ctx_char_embed],dim=2)
        
        ques_contextual_inp = torch.cat([ques_word_embed, ques_char_embed],dim=2)
        
        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)

        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1,1,ques_len,1)
        
        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1,ctx_len,1,1)
        
        elementwise_prod = torch.mul(ctx_, ques_)
        
        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        
        a = F.softmax(similarity_matrix, dim=-1)
        
        c2q = torch.bmm(a, ques_contextual_emb)
        
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        b = b.unsqueeze(1)
        q2c = torch.bmm(b, ctx_contextual_emb)
        q2c = q2c.repeat(1, ctx_len, 1)
        
        G = torch.cat([ctx_contextual_emb, c2q, 
                       torch.mul(ctx_contextual_emb,c2q), 
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)
        
        M, _ = self.modeling_lstm(G)
        
        M2, _ = self.end_lstm(M)
        
        p1 = self.output_start(torch.cat([G,M], dim=2))
        
        p1 = p1.squeeze()
        
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        
        return p1, p2
    
CHAR_VOCAB_DIM = len(char2idx)
EMB_DIM = 300
CHAR_EMB_DIM = 8
NUM_OUTPUT_CHANNELS = 100
KERNEL_SIZE = (8,5)
HIDDEN_DIM = 100
device = torch.device('cuda')

model = BiDAF(CHAR_VOCAB_DIM, 
              EMB_DIM, 
              CHAR_EMB_DIM, 
              NUM_OUTPUT_CHANNELS, 
              KERNEL_SIZE, 
              HIDDEN_DIM, 
              device).to(device)


optimizer = optim.Adadelta(model.parameters())

def train(model, train_dataset):
    print("Starting training ........")
   

    train_loss = 0.
    model.train()
    for batch in tqdm(train_dataset):
        
        optimizer.zero_grad()

        context, question, char_ctx, char_ques, label, ctx_text, ans, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device),\
                                   char_ctx.to(device), char_ques.to(device), label.to(device)


        preds = model(context, question, char_ctx, char_ques)

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

        context, question, char_ctx, char_ques, label, ctx, answers, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device),\
                                   char_ctx.to(device), char_ques.to(device), label.to(device)
        
       

        
        with torch.no_grad():
            
            s_idx, e_idx = label[:,0], label[:,1]

            preds = model(context, question, char_ctx, char_ques)

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
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    start_time = time.time()
    
    train_loss = train(model, train_dataset)
    valid_loss, em, f1 = valid(model, valid_dataset)  

    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
        'em':em,
        'f1':f1,
        }, 'model/bidaf_run4_{}.pth'.format(epoch+1))
    
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

e = [i for i in range(1,11)]

plt.figure()
plt.plot(e, train_losses, marker='o', label='Train Loss')
plt.plot(e, valid_losses, marker='s', label='Valid Loss')
plt.title('Loss vs epochs')
plt.xlabel('Loss')
plt.ylabel('epochs')
plt.legend()
plt.show()


plt.figure()
plt.plot(e, ems, marker='o', label='EM')
plt.plot(e, f1s, marker='s', label='F1')
plt.title('Score vs epochs')
plt.xlabel('Score')
plt.ylabel('epochs')
plt.legend()
plt.show()
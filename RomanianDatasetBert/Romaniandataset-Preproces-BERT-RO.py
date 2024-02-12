#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install transformers\n')


# In[39]:


import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel , AutoModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import unicodedata
import string
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib


# In[3]:


device = torch.device('cpu') 
data_path   = '/Users/catalina.chirita/Desktop/Disertatie/RomanianSatireDataset/'
train_path = data_path + "train.csv"
test_path = data_path + "test.csv"
val_path = data_path + "validation.csv"


# In[4]:


def createDataset(data_path, path_texts):
    dataset = pd.read_csv(path_texts, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    dataset['label']= dataset['label'].apply(lambda x:'satire' if x==1 else 'nonSatire')
    dataset.dropna(subset=['content'],inplace = True)
    dataset.reset_index(drop=True, inplace=True)
    dataset.drop(columns = ['title','index'], axis =1 , inplace = True)
    return dataset

pd_train = createDataset(data_path,train_path)
pd_test = createDataset(data_path,test_path)
pd_val = createDataset(data_path,val_path)


# In[7]:


pd_train['content'][1]


# In[6]:


pd_train = pd_train.sample(n=8000) 
pd_train.reset_index(drop=True, inplace=True)
pd_test.reset_index(drop=True, inplace=True)
print(pd_train)
print(pd_test)


# In[7]:


satire = 0
nonsatire=0
for array in range(len(pd_train["content"])):
    if pd_train["label"][array] == "satire":
        satire= satire+1
    else:
        nonsatire=nonsatire+1


# In[8]:


print(satire)
print(nonsatire)


# In[9]:


for array in range(len(pd_train["content"])):
    pd_train["content"][array]=unicodedata.normalize('NFD', pd_train["content"][array]).encode('ascii', 'ignore').decode("utf-8")
     
for array in range(len(pd_test["content"])):
    pd_test["content"][array]=unicodedata.normalize('NFD', pd_test["content"][array]).encode('ascii', 'ignore').decode("utf-8")
    
for array in range(len(pd_val["content"])):
    pd_val["content"][array]=unicodedata.normalize('NFD', pd_val["content"][array]).encode('ascii', 'ignore').decode("utf-8")    
    


# In[10]:


pd_train['content'][0]


# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

pd_train['content']= pd_train['content'].apply(lambda x: word_tokenize(x))
pd_test['content']= pd_test['content'].apply(lambda x: word_tokenize(x))

final_stopwords_list = stopwords.words('romanian')
final_stopwords_list.append('«')
final_stopwords_list.append('»')
final_stopwords_list.append('’')
final_stopwords_list.append('-')
final_stopwords_list.append('$')
final_stopwords_list.append('NE')
final_stopwords_list.append('(')
final_stopwords_list.append(')')
def remove_stopwords(text):
    output= [i for i in text if i not in final_stopwords_list]
    return output
def remove_digit(text):
    output= [c for c in text if not c.isdigit()]
    return output

pd_train['content']= pd_train['content'].apply(lambda x:remove_stopwords(x))
pd_train['content']= pd_train['content'].apply(lambda x:remove_digit(x))

pd_test['content']= pd_test['content'].apply(lambda x:remove_stopwords(x))
pd_test['content']= pd_test['content'].apply(lambda x:remove_digit(x))

for array in range(len(pd_train["content"])):
    pd_train["content"][array]=' '.join(pd_train["content"][array]) 
    
for array in range(len(pd_test["content"])):
    pd_test["content"][array]=' '.join(pd_test["content"][array]) 


# In[9]:


pd_test['content'][0]


# In[12]:


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
pd_train['content']= pd_train['content'].apply(lambda x:remove_punctuation(x))
pd_train['content']= pd_train['content'].apply(lambda x: x.lower())

pd_test['content']= pd_test['content'].apply(lambda x:remove_punctuation(x))
pd_test['content']= pd_test['content'].apply(lambda x: x.lower())


# In[13]:


pd_test['content'][0]


# In[14]:


pd_train.to_pickle("romaniandataset.pkl")
pd_test.to_pickle("testromaniandataset.pkl")


# In[15]:


np.random.seed(112)
df_train, df_val, df_test = np.split(pd_train.sample(frac=1, random_state=42), 
                                     [int(.8*len(pd_train)), int(.9*len(pd_train))])

print(len(df_train),len(df_val), len(df_test))


# In[40]:


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        #embedding vectors of all of the tokens in a sequence + embedding vector of [CLS] token
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer , pooled_output
        


# In[41]:


from transformers import AutoTokenizer, AutoModel
import torch
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

labels = {'nonSatire':0,
          'satire':1
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    


# In[16]:


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  


# In[42]:


from sklearn.ensemble import VotingClassifier
import pandas as pd
    
def evaluate2models2(model,test_dataset):

    test = Dataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    dataframe_val = []
    dataframe_wght = []
    dataframe_satire=[]
    dataframe_label = []
    dataframe_cls =[]
    pooled_output_dat = []
    with torch.no_grad():

    
        for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                pooled_output = output[1]
                print("output", output[0])
                print("pooled-output", pooled_output)
                acc = (output[0].argmax(dim=1) == test_label).sum().item()
                if acc == 1:
                    if output[0].argmax(dim=1)[0] == 1:
                        dataframe_satire.append('satire')
                        print('satire')
                    else:
                        dataframe_satire.append('nonSatire')
                        print('nonSatire')
                else:
                    dataframe_satire.append('error')
                    print('error of preddiction')
                    
                pooled_output = pooled_output.tolist()
                pooled_output_dat.append(pooled_output)
                output = output[0].tolist()  
                test_label = test_label.tolist()
                dataframe_val.append(output[0])
                dataframe_wght.append(float(0.60))
                dataframe_label.append(test_label[0])
                total_acc_test += acc
    
        model_dataset = {'Value':dataframe_val ,
        'Weight': dataframe_wght,
        'Label': dataframe_label,
        'Type': dataframe_satire}
        
        model_dataset = pd.DataFrame(model_dataset)
        model_dataset.to_pickle("dataframero.pkl")
        
        model_d = {'Pooled output romanian':pooled_output_dat}
 
        model_dataset2 = pd.DataFrame(model_d)
        model_dataset2.to_pickle("dataframero_pldout.pkl")
        
        print("Model1 DF", model_dataset)
        print(f'Test Accuracy model: {total_acc_test / len(test_dataset): .3f}')


# In[20]:


EPOCHS = 5
model = BertClassifier()
LR = 1e-6
              
train(model, df_train, df_val, LR, EPOCHS)


# In[21]:


filename = 'finalized_model_rom.sav'
joblib.dump(model, filename)


# In[8]:


device = torch.device('cpu') 
data_path   = '/Users/catalina.chirita/Desktop/Disertatie/RomanianSatireDataset/'
test_path = data_path + "test.csv"
pd_test = createDataset(data_path,test_path)


# In[43]:


filename = 'finalized_model_rom.sav'
loaded_model = joblib.load(filename)

pd_test = pd.read_pickle("testromaniandataset.pkl")


# In[44]:


evaluate2models2(loaded_model, pd_test)


# In[26]:


pd_train = pd.read_pickle("romaniandataset.pkl")
pd_train_eng = pd.read_pickle("romaniantoenglish.pkl")

pd_test = pd.read_pickle("testromaniandataset.pkl")
pd_test_eng = pd.read_pickle("testromaniantoenglish.pkl")


# In[27]:


pd_train


# In[28]:


pd_train_eng


# In[29]:


pd_test_eng


# In[30]:


pd_test


# In[31]:


pd_test["content"][0]


# In[32]:


pd_test_eng["content"][0]


# In[ ]:





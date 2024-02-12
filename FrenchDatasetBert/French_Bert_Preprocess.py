#!/usr/bin/env python
# coding: utf-8

# In[67]:


#!pip install spacy-lefff
get_ipython().system('pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git&> /dev/null')


# In[1]:


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
import os 
import string 
import glob


# In[2]:


device = torch.device('cpu') 
data_path_train   = '/Users/catalina.chirita/Desktop/Disertatie/FrenchSatireDataset/train/'
texts_path_train = "/Users/catalina.chirita/Desktop/Disertatie/FrenchSatireDataset/train/texts/*.txt"

data_path_test   = '/Users/catalina.chirita/Desktop/Disertatie/FrenchSatireDataset/test/'
texts_path_test = "/Users/catalina.chirita/Desktop/Disertatie/FrenchSatireDataset/test/texts/*.txt"

def createDataset(path, path_texts):
    train_label_path = path + "summary.tsv"
    df_train_label = pd.read_csv(train_label_path, delimiter="\t")
    txt_files = glob.glob(path_texts)
    txt_files.sort()
    data = []
    for files in txt_files:
        with open(files) as content:
            i=0
            for item in content:
                if i == 0:
                    interm = item
                    i = i+1
                else:
                    interm= interm + item
                    i= i+1
            data.append(interm)

    dataframe = {'content': data,
                'label': df_train_label['label']
                }

    frenchds = pd.DataFrame(dataframe)
    frenchds['label']= frenchds['label'].apply(lambda x:'satire' if x==1 else 'nonSatire')
    frenchds['content'] = frenchds['content'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    frenchds['content'] = frenchds['content'].replace("\\n", "")
    return frenchds


# In[3]:


train_dataset = createDataset(data_path_train,texts_path_train)
test_dataset=createDataset(data_path_test,texts_path_test)


# In[4]:


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
train_dataset['content']= train_dataset['content'].apply(lambda x:remove_punctuation(x))
train_dataset['content']= train_dataset['content'].apply(lambda x: x.lower())


test_dataset['content']= test_dataset['content'].apply(lambda x:remove_punctuation(x))
test_dataset['content']= test_dataset['content'].apply(lambda x: x.lower())


# In[5]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

train_dataset['content']= train_dataset['content'].apply(lambda x: word_tokenize(x))
test_dataset['content']= test_dataset['content'].apply(lambda x: word_tokenize(x))

final_stopwords_list = stopwords.words('french')
final_stopwords_list.append('«')
final_stopwords_list.append('»')
final_stopwords_list.append('’')
final_stopwords_list.append('-')
def remove_stopwords(text):
    output= [i for i in text if i not in final_stopwords_list]
    return output
def remove_digit(text):
    output= [c for c in text if not c.isdigit()]
    return output

train_dataset['content']= train_dataset['content'].apply(lambda x:remove_stopwords(x))
train_dataset['content']= train_dataset['content'].apply(lambda x:remove_digit(x))

test_dataset['content']= test_dataset['content'].apply(lambda x:remove_stopwords(x))
test_dataset['content']= test_dataset['content'].apply(lambda x:remove_digit(x))

for array in range(len(train_dataset["content"])):
    train_dataset["content"][array]=' '.join(train_dataset["content"][array]) 
    
for array in range(len(test_dataset["content"])):
    test_dataset["content"][array]=' '.join(test_dataset["content"][array]) 


# In[7]:


train_dataset.head()


# In[32]:


train_dataset.to_pickle("francedataset.pkl")
test_dataset.to_pickle("testfrancedataset.pkl")


# In[22]:


np.random.seed(112)
df_train, df_val, df_test = np.split(train_dataset.sample(frac=1, random_state=42), 
                                     [int(.8*len(train_dataset)), int(.9*len(train_dataset))])

print(len(df_train),len(df_val), len(df_test))


# In[2]:


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("cmarkea/distilcamembert-base")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer, pooled_output


# In[3]:


from transformers import AutoTokenizer, AutoModel
import torch
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base")

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
    


# In[12]:


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
                  


# In[13]:


EPOCHS = 5
model = BertClassifier()
LR = 1e-6
              
train(model, df_train, df_val, LR, EPOCHS)


# In[14]:


# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

filename = 'finalized_model_french.sav'
joblib.dump(model, filename)


# In[17]:


evaluate(model, df_test)


# In[10]:


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
        model_dataset.to_pickle("dataframefr.pkl")
        
        model_d = {'Pooled output france':pooled_output_dat}
 
        model_dataset2 = pd.DataFrame(model_d)
        model_dataset2.to_pickle("dataframefr_pldout.pkl")
        
        print("Model1 DF", model_dataset)
        print(f'Test Accuracy model: {total_acc_test / len(test_dataset): .3f}')


# In[11]:


import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

filename = 'finalized_model_french.sav'
loaded_model = joblib.load(filename)
test_dataset = pd.read_pickle("testfrancedataset.pkl")

evaluate2models2(loaded_model,test_dataset)


# In[11]:


dataset1 = pd.read_pickle("dataframefr.pkl")
dataset2 = pd.read_pickle("dataframefr-eng2.pkl")


# In[12]:


dataset1


# In[13]:


dataset2


# In[20]:


import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set(rc={'figure.figsize':(12,8)})
france = []
argmax = []
total_acc =0

for i in range(len(dataset1)):
    francedat = np.array(dataset1["Value"][i]) * dataset1["Weight"][i] 
    france_engdat = np.array(dataset2["Value"][i]) * dataset2["Weight"][i]
    
    temp = [francedat + france_engdat]
    france += temp
    argmax.append(np.argmax(temp))
    acc = np.argmax(temp)
    total_acc += acc
    
model_dataset = {'Ensemble': france,
         'ArgMax': argmax}

dataset = pd.DataFrame(model_dataset)
    
# print(dataset)
# sns.scatterplot(x="French",y="French-English",data=dataset,hue="ArgMax");
print(f'Test Accuracy model: {total_acc / len(dataset1): .3f}')


# In[ ]:


dataset


# In[ ]:






# coding: utf-8

# In[83]:

import pandas as pd
import numpy as np
import csv
import sys
import keras
from keras.preprocessing.text import one_hot 


# In[2]:

word_embeddings=pd.read_csv('vocab.csv')


# In[38]:

text=open('Sherelok Holmes.txt', 'r')


words=[]
for line in text:
    word=line.split()
    words=words+(word)

words_txt=[word.lower() for word in words]
print ("Before {}".format(len(words)))
words=list(set(words_txt))

word_df=pd.DataFrame(words)
word_df.to_csv('words.csv', header=False, index=False)

print ("Total unique words found {}".format(len(words)))

words_to_int=dict((c,i) for i, c in enumerate(words))
print (words_to_int)


# In[9]:

'''from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=5000)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index'''


# In[4]:

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[7]:

embedding_matrix = np.zeros((len(words_to_int) + 1, 100))
for i, word in enumerate(words):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[64]:

#embedding_matrix has all the words in the sherelok holmes text mapped to the glove embedding matrix
embedding_matrix.shape


# In[30]:

words_to_int['the']


# In[67]:

#Making sequneces of words of length 5 to be fed into our Model
seq_length=5
X=[]
y=[]

for i in range(0, len(words)-seq_length):
    seq_in=words_txt[i:i+seq_length]
    seq_out=words_txt[i+seq_length]
    X.append([word for word in seq_in])
    y.append(seq_out)


# In[146]:

#Same sequence as above with indexed as numbers
seq_length=5
X_num=[]
y_num=[]

for i in range(0, len(words)-seq_length):
    seq_in=words_txt[i:i+seq_length]
    seq_in=[words_to_int[k] for k in seq_in]
    seq_out=words_txt[i+seq_length]
    seq_out=words_to_int[seq_out]
    X_num.append([word for word in seq_in])
    y_num.append(seq_out)


# In[74]:

seq_embedding=np.zeros([len(X_num),5,100],dtype=float)


# In[77]:

#seq_embedding=np.zeros([len(X_num),5,100],dtype=float)
for i in range(0,len(X_num)):
    for j in range(0,len(X_num[0])):
        seq_embedding[i][j]=embedding_matrix[X_num[i][j]]


# In[98]:

y_num=keras.utils.np_utils.to_categorical(y_num,nb_classes=14031)


# In[104]:

train_input=seq_embedding[0:12000]
test_input=seq_embedding[12000:]
train_output=y_num[0:12000]
test_output=y_num[12000:]


# In[105]:

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda,GRU,Embedding
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Merge


# In[131]:

model=Sequential()


# In[132]:

model.add(GRU(output_dim=32,return_sequences=False,go_backwards=False,init='he_normal',input_shape=(5,100)))


# In[133]:

model.add(Dense(14031,activation='softmax'))


# In[141]:

model.compile(optimizer='rmsprop',loss=keras.objectives.categorical_crossentropy,metrics=['accuracy'])


# In[142]:

model.output_shape


# In[143]:

model.fit(train_input,train_output,batch_size=128,nb_epoch=1,validation_data=[test_input,test_output])


# In[140]:

train_input.shape


# In[144]:

np.save('/home/saurabh/Desktop/train_input.npy',train_input)


# In[145]:

np.save('/home/saurabh/Desktop/test_input.npy',test_input)
np.save('/home/saurabh/Desktop/train_output.npy',train_output)
np.save('/home/saurabh/Desktop/test_output.npy',test_output)


# In[147]:

np.save('/home/saurabh/Desktop/y.npy',y_num)


# In[ ]:




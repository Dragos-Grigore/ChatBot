#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import numpy as np


# In[3]:


with open('train_qa.txt','rb')as f:
    train_data=pickle.load(f)


# In[4]:


with open('test_qa.txt','rb')as f:
    test_data=pickle.load(f)


# In[5]:


print(len(train_data),len(test_data))


# In[6]:


' '.join(train_data[0][0])


# In[7]:


' '.join(train_data[0][1])


# In[8]:


all_data=test_data+train_data


# In[9]:


vocab=set()

for story,question,answer in all_data:
    vocab=vocab.union(set(story))
    vocab=vocab.union(set(question))


# In[10]:


vocab.add('yes')


# In[11]:


vocab.add('no')


# In[12]:


vocab


# In[13]:


vocab_len=len(vocab)+1


# In[14]:


vocab_len


# In[15]:


all_story_lens=[len(data[0]) for data in all_data]


# In[16]:


max_story_len=max(all_story_lens)


# In[17]:


max_question_len=max([len(data[1]) for data in all_data])


# In[18]:


max_question_len


# In[19]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[20]:


tokenizer=Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)


# In[21]:


tokenizer.word_index


# In[22]:


train_story_text=[]
train_question_text=[]
train_answers=[]


# In[23]:


for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)


# In[24]:


train_story_text


# In[25]:


train_story_seq=tokenizer.texts_to_sequences(train_story_text)


# In[26]:


len(train_story_seq)


# In[27]:


len(train_story_text)


# In[28]:


train_story_seq


# In[29]:


def vectorize_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_question_len=max_question_len):
    
    X=[]
    Xq=[]
    Y=[]
    
    for story,query,answer in data:
        
        x=[word_index[word.lower()]for word in story]
        xq=[word_index[word.lower()]for word in query]
        
        y=np.zeros(len(word_index)+1)
        
        y[word_index[answer]]=1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_question_len),np.array(Y))


# In[30]:


inputs_train,queries_train,answer_train=vectorize_stories(train_data)


# In[31]:


inputs_test,queries_test,answer_test=vectorize_stories(test_data)


# In[32]:


from keras.models import Sequential,Model


# In[33]:


from keras.layers.embeddings import Embedding


# In[34]:


from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM


# In[35]:


input_sequence=Input((max_story_len,))
question=Input((max_question_len,))


# In[36]:


vocab_size=len(vocab)+1


# In[45]:


input_encoder_m=Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))


# In[46]:


input_encoder_c=Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[47]:


question_encoder=Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))


# In[55]:


input_encoded_m=input_encoder_m(input_sequence)
input_encoded_c=input_encoder_c(input_sequence)
question_encoded=question_encoder(question)


# In[56]:


match=dot([input_encoded_m,question_encoded],axes=(2,2))
match=Activation('softmax')(match)


# In[57]:


response=add([match,input_encoded_c])
response=Permute((2,1))(response)


# In[58]:


answer=concatenate([response,question_encoded])


# In[59]:


answer


# In[60]:


answer=LSTM(32)(answer)


# In[62]:


answer=Dropout(0.5)(answer)
answer=Dense(vocab_size)(answer)


# In[63]:


answer=Activation('softmax')(answer)


# In[65]:


model=Model([input_sequence,question],answer)


# In[66]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[67]:


model.summary()


# In[68]:


history=model.fit([inputs_train,queries_train],answer_train,batch_size=32,epochs=3,validation_data=([inputs_test,queries_test],answer_test))


# In[71]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[87]:


model.save('mynewmodel.h5')


# In[88]:


model.load_weights('chatbot_10.h5')


# In[89]:


pred_results=model.predict(([inputs_test,queries_test]))


# In[90]:


test_data[0][0]


# In[91]:


pred_results[0]


# In[92]:


val_max=np.argmax(pred_results[0])


# In[93]:


for key,val in tokenizer.word_index.items():
    if val==val_max:
        k=key


# In[94]:


k


# In[95]:


pred_results[0][val_max]


# In[97]:


my_story="John left the kitchen . Sandra dropped the football in the garden ."


# In[98]:


my_story.split()


# In[99]:


my_question="Is the football in the garden ."


# In[100]:


my_question.split()


# In[101]:


mydata=[(my_story.split(),my_question.split(),'yes')]


# In[102]:


mydata


# In[103]:


my_story,my_ques,my_ans=vectorize_stories(mydata)


# In[104]:


my_ans


# In[105]:


pred_results=model.predict(([my_story,my_ques]))


# In[106]:


val_max=np.argmax(pred_results[0])


# In[107]:


for key,val in tokenizer.word_index.items():
    if val==val_max:
        k=key


# In[108]:


k


# In[ ]:





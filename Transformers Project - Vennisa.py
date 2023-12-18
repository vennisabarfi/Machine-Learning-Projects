#!/usr/bin/env python
# coding: utf-8

# In[25]:


pip install transformers


# In[4]:


pip install torch


# In[5]:


pip install tensorflow


# In[6]:


pip install flax


# In[7]:


pip install ml-dtypes==0.2.0


# In[8]:


pip install ml-dtypes==0.3.1


# In[1]:


pip install --upgrade tensorflow-intel


# In[1]:


from transformers import pipeline


# In[9]:


# Model Approach: Question-Answering
qa_1 = pipeline(model="deepset/roberta-base-squad2") #specific model used from deepset
qa_1(question="What is my name?", context="My name is Vennisa, I live in San Antonio and I am from Ghana")

# Model Source: https://huggingface.co/roberta-base 


# In[14]:


# Task Approach: Question-Answering
qa_model = pipeline("question-answering") #task 
question = "Where do I live"
context = "My name is Vennisa and I live in San Antonio"
qa_model (question = question, context = context)


# In[15]:


# Model Approach: Summarization
summarizer = pipeline("summarization", model = "Falconsai/text_summarization")

text =  """ Yoruba is a language spoken in West Africa, primarily in Southwestern and Central Nigeria. It is spoken by the ethnic Yoruba people. The number of Yoruba speakers is roughly 44 million, plus about 2 million second-language speakers.[1] As a pluricentric language, it is primarily spoken in a dialectal area spanning Nigeria, Benin, and Togo with smaller migrated communities in Côte d'Ivoire, Sierra Leone and The Gambia.

Yoruba vocabulary is also used in the Afro-Brazilian religion known as Candomblé, in the Caribbean religion of Santería in the form of the liturgical Lucumí language and in various Afro-American religions of North America. Most modern practitioners of these religions in the Americas do not actually speak or understand the Yoruba language, rather they use Yoruba words and phrases for songs that for them are incomprehensible. Usage of a lexicon of Yoruba words and short phrases during ritual is also common, but they have gone through changes due to the fact that Yoruba is no longer a vernacular for them and fluency is not required.[4][5][6][7]

As the principal Yoruboid language, Yoruba is most closely related to the languages Itsekiri (spoken in the Niger Delta), and Igala (spoken in central Nigeria).

"""

print(summarizer(text, max_length =200, min_length = 50, do_sample = False))

# Text Source: Wikipedia - https://en.wikipedia.org/wiki/Yoruba_language
# Model Source: 


# In[16]:


# introducing some randomness 
summarizer = pipeline("summarization", model = "Falconsai/text_summarization")

text =  """ Yoruba is a language spoken in West Africa, primarily in Southwestern and Central Nigeria. It is spoken by the ethnic Yoruba people. The number of Yoruba speakers is roughly 44 million, plus about 2 million second-language speakers.[1] As a pluricentric language, it is primarily spoken in a dialectal area spanning Nigeria, Benin, and Togo with smaller migrated communities in Côte d'Ivoire, Sierra Leone and The Gambia.

Yoruba vocabulary is also used in the Afro-Brazilian religion known as Candomblé, in the Caribbean religion of Santería in the form of the liturgical Lucumí language and in various Afro-American religions of North America. Most modern practitioners of these religions in the Americas do not actually speak or understand the Yoruba language, rather they use Yoruba words and phrases for songs that for them are incomprehensible. Usage of a lexicon of Yoruba words and short phrases during ritual is also common, but they have gone through changes due to the fact that Yoruba is no longer a vernacular for them and fluency is not required.[4][5][6][7]

As the principal Yoruboid language, Yoruba is most closely related to the languages Itsekiri (spoken in the Niger Delta), and Igala (spoken in central Nigeria).

"""

print(summarizer(text, max_length =200, min_length = 50, do_sample = True)) #set to True to introduce randomness

# Produced some errors with 'Yoruboid' but more concise summary


# In[23]:


# Simple Machine Translation (English to German)

text ="translate English to French: I am currently working on this translation project using transformers for the first time. I am so excited "
trans = pipeline("translation", model = "t5-small")
trans(text)


# In[24]:


# Using ABENA the Twi Machine Translation Model
MODEL= "Ghana-NLP/distilabena-base-akuapem-twi-cased"

twi_trans = pipeline("fill-mask", model = MODEL, tokenizer = MODEL)

print(twi_trans("Me din [MASK] Vennisa. "))

#Review: It's actually a pretty good fillmask. Me din de Vennisa - My name is Vennisa is the correct output.


# In[ ]:





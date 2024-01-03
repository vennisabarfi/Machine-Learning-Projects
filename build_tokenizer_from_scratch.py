#!/usr/bin/env python
# coding: utf-8

# Tokenization is comprised of several steps:
# 1. Normalization: Cleaning up text that is deemed necessary ( involves removing , spaces or accents).
# 2. Pre-tokenization (splitting the input into words)
# 3. Running the input through the model. This uses pre-tokenized words to produce a sequence of tokens.
# 4. Post-processing. This is adding the special tokens of the tokenizer, generating the attention mask and token type IDs.
# 
# Link to helpful documentation for each of the 4 listed above: https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers

# In[1]:


# Acquiring a corpus
# We are using the WikiText2 dataset 

from datasets import load_dataset

dataset = load_dataset("wikitext", name ="wikitext-2-raw-v1", split="train") #load just the training split of the dataset


# In[2]:


# break the training data down and yield/deliver in chunks of 1000 
def get_training_corpus():
    for i in range(0, len(dataset), 1000): # 0 to dataset length and step-size of 1000
        yield dataset[i: i+ 1000]["text"] # slices text column of the dataset and generates chunks of the training data in batches of 1000
        


# In[3]:


# How to train a tokenizer on text files directly (optional)
# create a text file containing all the texts/inputs from WikiText-2 that we can use locally
with open("wikitext-2.txt", "w", encoding = "utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")


# Build a WordPiece tokenizer from scratch

# In[4]:


from tokenizers import(
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer

)

tokenizer = Tokenizer(models.WordPiece(unk_token = "[UNK]"))


# In[5]:


# Normalization using BertNormalizer with classic options: lowercase and strip_accents
# this will replicate the bert-base-uncased tokenizer

tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)


# In[6]:


# Now building a BERT normalizer by hand
# Building a normalizer using a Sequence of several normalizers to build our own custom one. Order is important
# NFD Unicode normalize works hand in hand with StripAccents to properly recognize the accented characters
tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])


# In[7]:


# to check out the effects of normalizer in a string output use
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))


# Pre-Tokenization Step

# In[8]:


# using the BertPreTokenizer
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# now test the output
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer")
# output shows each word/token and its start and end indices


# In[9]:


# or from scratch use this which splits on whitespace and all characters that are not letters, digits, or the underscore character
pre_tokenizer = pre_tokenizers.WhitespaceSplit()
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")


# In[10]:


# You can also use Sequence to compose several pre-tokenizers

pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer")


# Training the Model using a WordPiece Trainer (in this case)

# In[11]:


# include the special tokens, to include them in the vocabulary. For this training corpus, they are not present
special_tokens = ["[UNK]","[PAD]","[CLS]", "[SEP]", "[MASK]"] 
#CLS - Start of the sequence, SEP - separator or separating segments in the text


# In[12]:


trainer = trainers.WordPieceTrainer(vocab_size =25000, special_tokens = special_tokens)
# other optional parameters include min_frequency - the number of times a token must appear to be included in the vocabulary
# continuining_subword_prefix - if we want to use something different from ##
#vocab_size — The size of the final vocabulary, including all tokens and alphabet.


# In[13]:


# Train the model using the iterator defined earlier

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


# In[16]:


# OPTIONAL
# Second option if we wanted to use text files to train our tokenizer
tokenizer.model = models.WordPiece (unk_token ="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)


# Test the Tokenizer on text using the encode() method

# In[14]:


encoding = tokenizer.encode("Let's get this tokenizer.")
print(encoding.tokens)


# Post-processing

# In[15]:


# Add the [CLS] and [SEP] tokens using TemplateProcessor(specify a template on how you are going to add these tokens to every sentence)
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id) # special ids for the tokens so the model can interpret them


# In[16]:


# Writing the template for the TemplateProcessor (how to treat a single sentence and a pair of sentences)
# Using the classic BERT template(see documentation)
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)


# In[17]:


# Now encoding
encoding = tokenizer.encode("Let's test this tokenizer")
print(encoding.tokens)


# In[18]:


# Now on a pair of sentences
encoding = tokenizer.encode("Let's test this tokenizer...", "And include a second sentence to test this out")
print(encoding.tokens)
print(encoding.type_ids)
# output 0 - tokens from first sentence, 1- tokens from second sentence


# Decoder

# In[19]:


tokenizer.decoder = decoders.WordPiece(prefix = "##")


# In[20]:


tokenizer.decode(encoding.ids)


# Save the Tokenizer in a single JSON file

# In[21]:


tokenizer.save("C:/Users/veyhn/PycharmProjects/HelloWorld/tokenizer.json")

#saved to local directory


# In[22]:


# Reload the file into a Tokenizer object 
new_tokenizer = Tokenizer.from_file("C:/Users/veyhn/PycharmProjects/HelloWorld/tokenizer.json")


# Use this Tokenizer in Transformers

# In[30]:


from transformers import PreTrainedTokenizerFast
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    tokenizer_file  = "C:/Users/veyhn/PycharmProjects/HelloWorld/tokenizer.json", #optional
    unk_token = "[UNK]",
    pad_token ="[PAD]",
    cls_token = "[CLS]",
    sep_token = "[SEP]",
    mask_token = "[MASK]",
)


# In[81]:


pip install transformers


# In[24]:


# you can also use a specific tokenizer class like BertTokenizerFast
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)


# In[31]:


# save the tokenizer
wrapped_tokenizer.save_pretrained("C:/Users/veyhn/PycharmProjects/HelloWorld")


# In[ ]:





import os
import urllib.request
import re
import tiktoken
from importlib.metadata import version

# just base on vocab can not recoginse new unknown word
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# tokenize
print("Total number of character:", len(raw_text))
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens:", len(preprocessed))
# get vocabulary index
all_words = sorted(set(preprocessed))
all_words.extend("<|endoftext|>")
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)
print("First 50 unique tokens:" + str(all_words[:10]))
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 10:
        break
#convert to tokenId base on vocab
tokenizer = SimpleTokenizerV1(vocab)
idLIst = tokenizer.encode(raw_text)
print("Tokenized text:", idLIst[:10])
testList = tokenizer.decode(idLIst)
print("Decoded text:", testList[:50])


print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))
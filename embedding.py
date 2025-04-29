import torch
import torch.nn as nn
from vocabulary import vocabulary


def create_numbered_vocab(word_list):
    return {word: idx + 1 for idx, word in enumerate(word_list)}

vocab = create_numbered_vocab(vocabulary)

vocab_size = len(vocab) + 1  # +1 for padding/unknown tokens
embedding_dim = 100 # Tweek depending on what we choose
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

sentence = "I will run right"
token_ids = [vocab[word] for word in sentence.split()]

input_tensor = torch.tensor([token_ids])
embeddings = embedding_layer(input_tensor)

print("Token IDs:", token_ids)
print("Embeddings:", embeddings)
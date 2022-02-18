import argparse
import numpy as np
import sys
import random

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab_pofo.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors_pofo.txt', type=str)
    parser.add_argument('--input_file', default='no file given', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    input_filepath = args.input_file



    return (W_norm, vocab, ivocab, input_filepath)


#this accounts for more than one word in `input_term`
def closest10(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            return([term]*10)
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:10]

    return [ivocab[x] for x in a]



def random_index(): 
    return (random.randint(1,10) - 1)


if __name__ == "__main__":
    W, vocab, ivocab, input_filepath = generate()
    while True:
        if input_filepath == 'no file given':

            input_term = input("\n\nEnter word or sentence without punctuation (EXIT to break): ")
            if input_term == 'EXIT':
                break
            else:
                all_words = input_term.lower().split(" ")
                transformed_words= [closest10(W, vocab, ivocab, a)[random_index()] for a in all_words]
                print("\n\n")
                print(" ".join(transformed_words))
        else:
            input_file = open(input_filepath, 'r')
            input_file_text = input_file.read()
            all_words = input_file_text.lower().split(" ")
            transformed_words= [closest10(W, vocab, ivocab, a)[random_index()] for a in all_words]
            input_file.close()
            print(" ".join(transformed_words))
            break



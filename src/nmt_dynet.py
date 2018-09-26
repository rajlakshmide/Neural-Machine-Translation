from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json

RNN_BUILDER = GRUBuilder

class nmt_dynet:

    def __init__(self, src_vocab_size, tgt_vocab_size, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word, word_d, gru_d, gru_layers):

        # initialize variables
        self.gru_layers = gru_layers
        self.src_vocab_size = src_vocab_size #size of src vocabulary
        self.tgt_vocab_size = tgt_vocab_size #size of trg vocabulary
        self.src_word2idx = src_word2idx #dict mapping between a word and its index - source
        self.src_idx2word = src_idx2word #dict mapping between an index and the word - source
        self.tgt_word2idx = tgt_word2idx #dict mapping between a word and its index - target
        self.tgt_idx2word = tgt_idx2word #dict mapping between a an index and its word - target
        self.word_d = word_d #input dimensionality?
        self.gru_d = gru_d #GRU dimensionality?

        self.model = Model()

        # the embedding paramaters
        self.source_embeddings = self.model.add_lookup_parameters((self.src_vocab_size, self.word_d)) #word embedding vectors for each word in src vocab; embedding is size word_d
        self.target_embeddings = self.model.add_lookup_parameters((self.tgt_vocab_size, self.word_d)) #word embedding vectors for each word in trg vocab; embedding is size word_d


        # YOUR IMPLEMENTATION GOES HERE
        # project the decoder output to a vector of tgt_vocab_size length
        self.output_w = self.model.add_parameters((self.tgt_vocab_size, self.gru_d))
        self.output_b = self.model.add_parameters(self.tgt_vocab_size)

        # encoder network
        # the foreword rnn
        self.fwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d/2, self.model)
        # the backword rnn
        self.bwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d/2, self.model)

        # decoder network
        self.dec_RNN = RNN_BUILDER(self.gru_layers, self.word_d+self.gru_d, self.gru_d, self.model)

        #gru = RNN_BUILDER(gru_layers, word_d, gru_d, &model)


    def encode(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return encoding of the source sentence
        '''
        # YOUR IMPLEMENTATION GOES HERE
        # NEED TO USE self.model.add_input() to process 1 word at a time
        # for _ in src_sentence:
        #   self.model.add_input(_)
        # need to get the hidden state (forward and backward) and concatenate, then pass the last one to decoder.
        #s = gru.initial_state()
        #states = []
        #for x in inputsL
        #s = s.add_input(x)
        #h = s.output()
        #states.append(h)
        s_forward = self.fwd_RNN.initial_state()
        forward_states = []
        for x in src_sentence:
            s_forward = s_forward.add_input(self.source_embeddings[self.src_word2idx[x]])
            h = s_forward.output()
            forward_states.append(h)
        s_backward = self.bwd_RNN.initial_state()
        backward_states = []
        for x in reversed(src_sentence):
            s_backward = s_backward.add_input(self.source_embeddings[self.src_word2idx[x]])
            h = s_backward.output()
            backward_states.append(h)
        h_v = concatenate([forward_states[-1],backward_states[-1]])
        #h_vlen = len(list(h_v))
        return h_v


        


    def get_loss(self, src_sentence, tgt_sentence):
        '''
        src_sentence: words in src sentence
        tgt_sentence: words in tgt sentence
        return loss for this source target sentence pair
        '''
        renew_cg()

        # YOUR IMPLEMENTATION GOES HERE
        v1 = vecInput(self.gru_d)
        encoding = self.encode(src_sentence)
        #v1.set(list(encoding))
        s0 = self.dec_RNN.initial_state()
        R = parameter(self.output_w)
        b = parameter(self.output_b)
        #want probability of target sentence given the source sentence
        s = s0
        loss = []
        for word, next_word in zip(tgt_sentence, tgt_sentence[1:]):
            s  = s.add_input(concatenate([self.target_embeddings[self.tgt_word2idx[word]], encoding]))
            probs = softmax(R*s.output() + b)
            loss.append(-log(pick(probs, self.tgt_word2idx[next_word])))
        loss = esum(loss)
        return loss
           

    def sample(self, probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    def generate(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return list of words in the target sentence
        '''
        renew_cg()
        R = parameter(self.output_w)
        b = parameter(self.output_b)

        # YOUR IMPLEMENTATION GOES HERE
        v1 = vecInput(self.gru_d)

        encoding = self.encode(src_sentence)
        #v1.set(list(encoding))
        s0 = self.dec_RNN.initial_state()
        predicted_output = []
        y = '<s>'
        predicted_output.append('<s>')
        s = s0
        while (True):
            s = s.add_input(concatenate([self.target_embeddings[self.tgt_word2idx[y]],encoding]))
            h = s.output()
            #y is the word in target vocab that maximizes softmax_y(f(s))
            probs = softmax(R*h+b)
            probs = probs.vec_value()
            #print (probs)
            #print (len(list(probs)))
            #probs = probs.vec_value()
            #next_word_index = self.sample(probs)
            highest_prob = float('-inf')
            i = None
            for index, probability in enumerate((probs)):
                #print(probability.value())
                if (probability > highest_prob):
                    highest_prob = probability
                    i = index
            next_word_index = i

            y = self.tgt_idx2word[next_word_index]
            predicted_output.append(y)
            if (y=='</s>'):
                break
        return predicted_output

    def translate_all(self, src_sentences):
        translated_sentences = []
        for src_sentence in src_sentences:
            # print src_sentence
            translated_sentences.append(self.generate(src_sentence))

        return translated_sentences

    # save the model, and optionally the word embeddings
    def save(self, filename):

        self.model.save(filename)
        embs = {}
        if self.src_idx2word:
            src_embs = {}
            for i in range(self.src_vocab_size):
                src_embs[self.src_idx2word[i]] = self.source_embeddings[i].value()
            embs['src'] = src_embs

        if self.tgt_idx2word:
            tgt_embs = {}
            for i in range(self.tgt_vocab_size):
                tgt_embs[self.tgt_idx2word[i]] = self.target_embeddings[i].value()
            embs['tgt'] = tgt_embs

        if len(embs):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(embs, f)

def get_val_set_loss(network, val_set):
        loss = []
        for src_sentence, tgt_sentence in zip(val_set.source_sentences, val_set.target_sentences):
            loss.append(network.get_loss(src_sentence, tgt_sentence).value())

        return sum(loss)

def main(train_src_file, train_tgt_file, dev_src_file, dev_tgt_file, model_file, num_epochs, embeddings_init = None, seed = 0):
    print('reading train corpus ...')
    train_set = Corpus(train_src_file, train_tgt_file)
    # assert()
    print('reading dev corpus ...')
    dev_set = Corpus(dev_src_file, dev_tgt_file)

    print ('Initializing simple neural machine translator:')
    # src_vocab_size, tgt_vocab_size, tgt_idx2word, word_d, gru_d, gru_layers
    encoder_decoder = nmt_dynet(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, 50, 50, 2)

    trainer = SimpleSGDTrainer(encoder_decoder.model)

    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    #print (sample_output)
    losses = []
    best_bleu_score = 0
    for epoch in range(num_epochs):
        print ('Starting epoch', epoch)
        # shuffle the training data
        combined = list(zip(train_set.source_sentences, train_set.target_sentences))
        random.shuffle(combined)
        train_set.source_sentences[:], train_set.target_sentences[:] = zip(*combined)

        print ('Training . . .')
        sentences_processed = 0
        for src_sentence, tgt_sentence in zip(train_set.source_sentences, train_set.target_sentences):
            loss = encoder_decoder.get_loss(src_sentence, tgt_sentence)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            sentences_processed += 1
            if sentences_processed % 4000 == 0:
                print ('sentences processed: ', sentences_processed)

        # Accumulate average losses over training to plot
        val_loss = get_val_set_loss(encoder_decoder, dev_set)
        print ('Validation loss this epoch', val_loss)
        losses.append(val_loss)

        print ('Translating . . .')
        translated_sentences = encoder_decoder.translate_all(dev_set.source_sentences)

        print('translating {} source sentences...'.format(len(sample_output)))
        for sample in sample_output:
            print('Target: {}\nTranslation: {}\n'.format(' '.join(dev_set.target_sentences[sample]),
                                                                         ' '.join(translated_sentences[sample])))

        bleu_score = get_bleu_score(translated_sentences, dev_set.target_sentences)
        print ('bleu score: ', bleu_score)
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            # save the model
            encoder_decoder.save(model_file)

    print ('best bleu score: ', best_bleu_score)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
#     parser.add_argument('model_type')
    parser.add_argument('train_src_file')
    parser.add_argument('train_tgt_file')
    parser.add_argument('dev_src_file')
    parser.add_argument('dev_tgt_file')
    parser.add_argument('model_file')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--dynet-mem')

    args = vars(parser.parse_args())
    args.pop('dynet_mem')

    main(**args)

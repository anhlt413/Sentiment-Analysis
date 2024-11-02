import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from vocab import *


class RNNModel(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim=50,
            embedding=None,
            use_embedding=True,
            num_layers = 1, 
            rnn_cell_class=nn.LSTM,
            hidden_dim=50,
            bidirectional=False,
            dropout = 0,
            freeze_embedding=False):

        super().__init__()
        self.vocab_size = vocab_size
        self.use_embedding = use_embedding
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.dropout = dropout 
        self.bidirectional = bidirectional
        self.freeze_embedding = freeze_embedding
        # Graph
        if self.use_embedding:
            self.embedding = self._define_embedding(
                embedding, vocab_size, self.embed_dim, self.freeze_embedding)
            self.embed_dim = self.embedding.embedding_dim
        self.rnn = rnn_cell_class(
            input_size=self.embed_dim,
            num_layers = self.num_layers, 
            hidden_size=hidden_dim,
            dropout  = self.dropout,
            batch_first=True,
            bidirectional=bidirectional)

    def forward(self, X, seq_lengths):
        if self.use_embedding:
            X = self.embedding(X)
        embs = torch.nn.utils.rnn.pack_padded_sequence(
            X,
            batch_first=True,
            lengths=seq_lengths.cpu(),
            enforce_sorted=False)
        outputs, state = self.rnn(embs)
        return outputs, state

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        elif isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding)
        else:
            return embedding


class RNNClassifierModel(nn.Module):
    def __init__(self, rnn, output_dim, classifier_activation, vocab):
        super().__init__()
        self.rnn = rnn
        self.vocab = vocab
        self.output_dim = output_dim
        self.hidden_dim = self.rnn.hidden_dim
        self.device = "cpu"
        if self.rnn.bidirectional:
            self.classifier_dim = self.hidden_dim * 2
        else:
            self.classifier_dim = self.hidden_dim
        self.hidden_layer = nn.Linear(
            self.classifier_dim, self.hidden_dim)
        self.classifier_activation = classifier_activation
        self.classifier_layer = nn.Linear(
            self.hidden_dim, self.output_dim)

    def forward(self, X, seq_lengths):
        X = self.vocab.to_input_tensor(X, self.device)
        outputs, state = self.rnn(X, seq_lengths)
        state = self.get_batch_final_states(state)
        if self.rnn.bidirectional:
            state = torch.cat((state[0], state[1]), dim=1)
        h = self.classifier_activation(self.hidden_layer(state))
        logits = self.classifier_layer(h)
        return logits

    def get_batch_final_states(self, state):
        if self.rnn.rnn.__class__.__name__ == 'LSTM':
            return state[0].squeeze(0)
        else:
            return state.squeeze(0)

    def predict(self, X, seq_lengths):
        logits = self.forward(X,seq_lengths)
        return logits.argmax(-1)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = RNNClassifierModel(vocab=params['vocab'], rnn = params['rnn'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path)

        params = {
            'args': dict(output_dim = self.output_dim, classifier_activation = self.classifier_activation),
            'vocab': self.vocab,
            'rnn': self.rnn,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

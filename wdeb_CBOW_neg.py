import torch
from torch.autograd import Variable
import torch.nn as nn

class word_embedding_network(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        """Initialize model parameters. Args: emb_size: Embedding size. emb_dimention: Embedding dimention, typically from 50 to 500. """
        #CBOWModeler
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """Forward process. As pytorch designed, all variables must be batch format, so all input of this method is a list of word id. Args: pos_u: list of center word ids for positive word pairs. pos_v: list of neibor word ids for positive word pairs. neg_v: list of neibor word ids for negative word pairs. """
        losses = []
        emb_u = self.u_embeddings(Variable(torch.LongTensor(pos_u)))
        emb_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)


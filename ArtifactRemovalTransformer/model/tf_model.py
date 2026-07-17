import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


# ---------------------Encoder and Decoder Stacks--------------------
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        #print("encode1:", src.shape)
        #temp = self.src_embed(src)
        #print("encode2:", temp.shape)
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        #print("decode1:", memory.shape)
        #print("decode2:", src_mask.shape)
        #print("decode3:", tgt.shape)
        #print("decode4:", tgt_mask.shape)

        #temp = self.tgt_embed(tgt)
        #print("decode5:", temp.shape)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        '''
        temp = self.proj(x)
        print("Generator1:", temp.shape)
        draw(3, temp[0, :, :], "Gen(lp)_" + str(2))
        temp = F.log_softmax(self.proj(x), dim=-1)
        print("Generator2:", temp.shape)
        draw(3, temp[0, :, :], "Gen(sm)_" + str(3))
        '''
        #return F.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)
        
# ---------------------Encoder--------------------
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2        


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #print("EncoderLayer2: ", x.shape)
        return self.sublayer[1](x, self.feed_forward)

# ---------------------Decoder--------------------
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #print("DecoderLayer1:", x.shape)
        #print("DecoderLayer2:", memory.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #print("DecoderLayer3:", src_mask.shape)
        #print("DecoderLayer4:", tgt_mask.shape)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        #print("DecoderLayer5:", tgt_mask.shape)
        x = self.sublayer[2](x, self.feed_forward)
        #print("DecoderLayer6:", tgt_mask.shape)
        #time.sleep(5)
        return x


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


#---------------------Attention--------------------        
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print("attention1:", query.shape, key.transpose(-2, -1).shape, value.shape, mask.shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print("attention2:", d_k, scores.shape)
    #draw(3, scores[0, 0, :, :], "scores_" + str(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #draw(3, scores[0, 0, :, :], "scores_" + str(2))
    p_attn = F.softmax(scores, dim = -1)
    #draw(3, p_attn[0, 0, :, :], "scores_" + str(3))
    if dropout is not None:
        p_attn = dropout(p_attn)
        #draw(3, p_attn[0, 0, :, :], "scores_" + str(4))
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        #print("MultiHeadedAttention1:", query.shape, key.shape, value.shape, mask.shape)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        '''
        print("MultiHeadedAttention2:", query.shape, key.shape, value.shape)
        for i in range(self.h):
            draw(3, query[0,i,:,:], "Query_"+str(i))
            draw(3, key[0, i, :, :], "Key_" + str(i))
            draw(3, value[0, i, :, :], "Value_" + str(i))
        '''
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        '''
        print("MultiHeadedAttention3:", x.shape, self.attn.shape)
        for i in range(self.h):
            draw(3, self.attn[0, i, :, :], "Atten map_" + str(i))
            draw(3, x[0, i, :, :], "X_" + str(i))
        '''
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        '''        
        print("MultiHeadedAttention4:", x.shape, self.linears[-1])
        draw(3, x[0, :, :], "Concat")

        temp = self.linears[-1](x)
        print("MultiHeadedAttention5:", temp.shape)
        draw(3, temp[0, :, :], "Linear Proj")
        '''
        return self.linears[-1](x)


# ---------------------Position-wise Feed-Forward Networks--------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        temp = self.w_1(x)
        print("PositionwiseFeedForward1:", temp.shape)
        draw(3, temp[0, :, :], "w_1")
        temp = F.relu(self.w_1(x))
        print("PositionwiseFeedForward2:", temp.shape)
        draw(3, temp[0, :, :], "w_1_relu")
        temp = self.dropout(F.relu(self.w_1(x)))
        print("PositionwiseFeedForward3:", temp.shape)
        draw(3, temp[0, :, :], "dropout")
        temp = self.w_2(self.dropout(F.relu(self.w_1(x))))
        print("PositionwiseFeedForward4:", temp.shape)
        draw(3, temp[0, :, :], "w_2")
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ---------------------Embeddings--------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #a = self.lut(x)
        #print("Embeddings:", a.shape)
        return self.lut(x) * math.sqrt(self.d_model)

# ---------------------ExpandConv--------------------
class ExpandConv(nn.Module):
    def __init__(self, d_model, vocab):
        super(ExpandConv, self).__init__()
        self.lut = nn.Conv1d(in_channels=vocab, out_channels=d_model, kernel_size=1)
        self.d_model = d_model

    def forward(self, x):
        #print("ExpandConv:", x.shape)
        convoluted_x = self.lut(x)
        convoluted_x = convoluted_x.permute(0, 2, 1)
        return convoluted_x * math.sqrt(self.d_model)
        
# ---------------------Positional Encoding--------------------
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # scaling term that decreases exponentially as the depth (i.e., column index in pe) increases.
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=128, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        #nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        nn.Sequential(ExpandConv(d_model, src_vocab), c(position)),
        nn.Sequential(ExpandConv(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

#tmp_model = make_model(10, 10, 2)
#print("all pass", tmp_model)

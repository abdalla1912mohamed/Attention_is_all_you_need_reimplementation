#### include libraries  
import torch 
import torch.nn  as nn 
import math
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA DSA

######## 
#implement the input embedding layer  , embedding is a vector of 512 size
class InputEmbeddings(nn.Module):
    def __init__(self,d_model : int, vocab_size:int):
        super().__init__() 
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        print(self.vocab_size)
        x = x.long()
        return self.embedding(x) * math.sqrt(self.d_model) 




class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int , dropout: float)-> None : 
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len # maximum sequence length 
        self.dropout = nn.Dropout(dropout) # regularization it was 0.1 in the paper 


        # create a matrix of shape ( seq_len , d_model-->512)
        pe = torch.zeros(seq_len, d_model) 
        # positional encoding matrix   PE = sin(pos/10000 ** 2i/d_model)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        ## apply the sin for even and cosin for odd 
        pe[:, 0::2] = torch.sin(position*div_term) ## all rows , even columns starting from 0 and increment by 2
        pe[:, 1::2] = torch.cos(position*div_term)  ## odd columns start from 1 increment by 2 
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + (self.pe[: , :x.shape[1] , :]).requires_grad_(False)
        ## this tensor is not learnt as it is constant 
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6 )-> None :
        super().__init__() 
        self.eps=eps 
        self.alpha = nn.Parameter(torch.ones(1))  # learning paramater multiplied 
        self.bias = nn.Parameter(torch.ones(1))  # learning paramater added

    def forward(self, x):  # x is the token
        if x is None:
            raise ValueError("Input tensor 'x' is None!")
    
    # Check if x is empty (shape is invalid)
        if x.size(0) == 0:
            raise ValueError("Input tensor 'x' is empty!")

    # Perform normalization
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
    
    # Return the normalized output
        return (self.alpha * (x - mean) / (std + self.eps) + self.bias)


    ## d_model = 512   d_ff (inner layer) = 2048 the feed forward layer 
    # FNN(x) = max(0,xW1 + b1)W2 + b2 two linear transformation RELU is activation function 


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int , d_ff : int , dropout : float) ->None : 
        super().__init__()
        self.linear_1 = nn.Linear(d_model , d_ff) # the first layer input and output #w1 and b1 
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff,d_model) # w2 and b2
    def forward(self , x) : 
        #(batch , seq_len , d_model) -- > (batch ,  seq_len , d_ff) --> (batch , seq_len , d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
#######################################################################################
####### MULTI HEAD ATTENTION ######################################
# ATTENTION(Q,K,V) = softmax(Q*kT/rootdk)*V 
# out input is transformed to a query , key , value 
# each K , Q , V is multiplied by Wq Wk Wv  then 
# then the matrix is splited to h matrices ( h is the number of heads )
# then it the reult in concatenated  

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h : int , dropout : float) -> None : 
        super().__init__() 
        self.d_model = d_model 
        self.h = h 
        assert d_model %h ==0 , " d_model is divisble by h the head number "
        self.d_k = d_model//h # each head partition 
        self.w_q = nn.Linear(d_model , d_model) 
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model) 
        self.w_o = nn.Linear(d_model , d_model)
        self.dropout = nn.Dropout(dropout) 
    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):
        d_k = query.shape[-1]  # extracting the last dimension of the query as it is a 3d tensor and d_model is the last 

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) 
        # transposing is necessary so each query is mulitplied by the column of all the keys 
        ## @ is matrix multiplication 
        if mask is not None : 
            attention_scores.masked_fill_(mask==0 , -1e9) 
        attention_scores = attention_scores.softmax(dim = -1) 
        # (batch , seq_len , seq_len) 
        if dropout is not None : 
            attention_scores = dropout(attention_scores) 
        
        return (attention_scores @ value) , attention_scores
            ## attentions scores is softmaxed then multiplied by the value matrix 
            ### we will visualize the attention scores later 
            ## @ is a  dot product 

    def forward(self,q,k,v,mask) : 
        # mask --> in case of a decoder , we replace it with a small value - infinity 
        query = self.w_q(q)  ## element wise ulitplicaation a scalar for each element 
        # a query is a 3d tensor 
        # first dimension is for the batch containing sequences of tokens 
        # the second dimension is of the sequence length 
        # the 3rd dimension is the mebedding per each token in the sequence of 512 column 
        key = self.w_k (k)  # (batch , seq_len , d_model ) --> (batch, seq_len , d_model)
        value = self.w_v (v)  
        # ( batch , seq_len ,d_model ) --> (batch , seq_len , h , d_k) --> ( batch , h ,seq_len , d_k)
        query = query.view(query.shape[0] , query.shape[1],self.h , self.d_k).transpose(1,2)
        key = key.view(key.shape[0] , key.shape[1],self.h , self.d_k).transpose(1,2)
        value = value.view(query.shape[0] , value.shape[1],self.h , self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) 
         # re transpose  (batch ,h, seq_len, d_k) --> (batch ,seqlen , h , d_k)--> (batch,seq_len, d_model)
        return self.w_o(x) 
        ## we have adjusted the dimension of the query
        ### visualize the shape and the matrix
        ### for example our data is batch = 2 , seq_len =3 , d_model = 512 
        # [   original query 
        # [     batch(0)             
        #[ 0.1, 0.2, 0.3, 0.4, 0.5 . .. 512 feature ]  seq1
        #[ 0.5, 0.6, 0.7, 0.8, 0.9]   
        #[ 0.1, 0.5, 0.35, 0.54, 0.35 ... .... 512 feature]  
        #[ 0.45, 0.36, 0.57, 0.668, 0.69] seqlen-1

        #] , [ seq 2 ........ batch(1)
        # ................
        ## ...................
        #]
        ### ---> after transformation we will seprate our shapae into 4 dimension 
        ## batch , head , seq_len , d_k   where d_k = d_model / h so if h = 6 
        ## we will have batch =2 , head = 6  , seq_len = 3 , d_k = 96 for each head
         # each batch contains a number of heads each heads contain a sequence of a certain length 
         # each head will focus upon d_k of embeddings to focus on 
class ResidualConnection(nn.Module) :
    def __init__(self, dropout : float) -> None : 
        super().__init__()
        self.dropout=nn.Dropout(dropout)
 
        self.norm = LayerNormalization() 
    def forward(self, x , sublayer) : 
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderBlock(nn.Module) : 
    def __init__(self,self_attention_block: MultiHeadAttentionBlock , feed_forward_block: FeedForwardBlock , dropout: float)-> None : 
        super().__init__()
        self.self_attention_block = self_attention_block 
        self.feed_forward_block = feed_forward_block 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) 
        ###########
    def forward(self,x,src_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask)) 
        x = self.residual_connections[1](x,self.feed_forward_block) 
        return x  

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__() 
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers : 
            x = layer(x,mask) 
        return self.norm(x)

class DecoderBlock(nn.Module) : 
    def __init__(self , self_attention_block: MultiHeadAttentionBlock , cross_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , dropout:float)-> None : 
        super().__init__() 
        self.self_attention_block = self_attention_block 
        self.cross_attention_block = cross_attention_block 
        self.feed_forward_block = feed_forward_block 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) 
    def forward(self,x,encoder_output, src_mask , tgt_mask): 
        # src mask is from the encoder source langauge 
        # tgt_mask coming from the decoder 
        x = self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x,lambda x : self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x,FeedForwardBlock) 
        return x 
    
class Decoder(nn.Module):
    def __init__(self , layers : nn.ModuleList) ->None :
        super().__init__()
        self.layers=layers
        self.norm = LayerNormalization() 
    def forward(self,x,encoder_output,src_mask,tgt_mask) :
        for layer in self.layers :
            x = layer(x,encoder_output,src_mask,tgt_mask) 
        return self.norm(x)  


class ProjectionLayer(nn.Module):
    def __init__(self,d_model : int, vocab_size : int )-> None : 
        super().__init__() 
        self.proj =nn.Linear(d_model,vocab_size) 
    def forward(self,x):
        return torch.log_softmax(self.proj(x), dim=-1) 

############################### buliding the transformer 
class Transformer(nn.Module):
    def __init__(self, encoder : Encoder , decoder : Decoder , src_embed : InputEmbeddings , tgt_embed: InputEmbeddings , src_pos : PositionalEncoding, tgt_pos : PositionalEncoding , projection_layer : ProjectionLayer ) :
        super().__init__() 
        self.encoder = encoder 
        self.decoder = decoder 
        self.projection_layer = projection_layer 
        self.src_embed = src_embed 
        self.tgt_embed = tgt_embed 
        self.src_pos = src_pos 
        self.tgt_embed = tgt_embed 

    def encode(self, src ,src_mask):
        src = self.src_embed(src) 
        src = self.src_pos(src) 
        return self.encoder(src,src_mask) 
    
    def decode(self,encoder_output , src_mask,tgt ,tgt_mask):
        tgt = self.tgt_embed(tgt) 
        tgt = self.tgt_embed(tgt) 
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask) 
    
    def project(self,x):
        return self.projection_layer(x) 
    
def build_transformer(src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int , d_model : int =512 , N : int = 6 , h : int = 8 , dropout : float = 0.1 , d_ff : int = 2048 ):
    # create the embedding layers 
    src_embed = InputEmbeddings(d_model , src_vocab_size) 
    tgt_embed = InputEmbeddings(d_model , tgt_vocab_size) 

    # create positional encoding 

    src_pos = PositionalEncoding(d_model , src_seq_len , dropout) 
    tgt_pos= PositionalEncoding(d_model , tgt_seq_len , dropout) 

    ## create encoder blocks 

    encoder_blocks = [ ] 
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model  ,h ,dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout) 
        encoder_block = EncoderBlock(encoder_self_attention_block , feed_forward_block, dropout) 
        encoder_blocks.append(encoder_block )
    
    ### decoder block 
    decoder_blocks = [ ] 
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model  ,h ,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model  ,h ,dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout) 
        decoder_block = DecoderBlock(decoder_self_attention_block ,decoder_cross_attention_block, feed_forward_block, dropout) 
        decoder_blocks.append(decoder_block ) 

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks)) 
    ## create the projection layer 
    projection_layer = ProjectionLayer(d_model , tgt_vocab_size) 


    ## create the transformer 
    transformer = Transformer(encoder , decoder ,src_embed , tgt_embed ,src_pos , tgt_pos, projection_layer) 

    #### intialize the paramters 

    for p in transformer.parameters() : 
        if p.dim()>1 : 
            nn.init.xavier_uniform_(p)
    return transformer 













    





         




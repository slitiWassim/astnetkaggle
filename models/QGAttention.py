import torch
from torch import nn

from grouped_query_attention_pytorch.attention import MultiheadGQA




class MultiheadGQAConv(nn.Module):
    def __init__(self, embedding_dim,channels):
        super().__init__()
        
        self.attention_layer = MultiheadGQA(embed_dim=embedding_dim, query_heads=4, kv_heads=2, device="cuda")
        self.convK = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=1,bias=False)
        self.convQ = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=2,stride=2,bias=False)
        self.convV = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=2,stride=2,bias=False)
        self.embedding_dim = embedding_dim

        


    def forward(self, x):
        k = self.convK(x)
        q = self.convQ(x)
        v = self.convV(x)
        




        x_shape = x.shape

        k = k.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        q = q.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        v = v.contiguous().view(x_shape[0],-1,self.embedding_dim) 
        output = self.attention_layer(k,q,v)[0]

        #output = self.norm(output)
        return output.view(x_shape)

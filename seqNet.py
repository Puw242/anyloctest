import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)
        # print("sequence feature shape",seqFt.shape)
        return seqFt
        

class cosine_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(cosine_seqNet, self).__init__()
        self.inDims = inDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # print("shape for cosine: ", x.shape) #[24, 5, 4096]
        for idx in range(len(x)):
            cosine_sim_x = x[idx]
            #print("cosine_sim_x shape after slice: ",cosine_sim_x.shape)
            cosine_sim_x = self.cosine_sim(cosine_sim_x)
            x[idx] = cosine_sim_x
        # print("x[0] after, ",x[0])
    
        x = x.permute(0,2,1) 
        # print("x before conv: ",x.shape)
        x = self.conv(x)
        
        x = torch.mean(x,-1)
        # print("after and mean conv: ",x.shape)
        return x

    def cosine_sim(self,sequence):
        # print("sequence type: ", type(sequence))
        num_frames = len(sequence)
        weights = []
        normalized_weights = np.zeros(num_frames)
        
        weighted_seq = []
        central_frame = sequence[(num_frames // 2) + 1].unsqueeze(0) #get central frame

        for j in range(num_frames):
           # print("cosine sim score: ", F.cosine_similarity(sequence[j].unsqueeze(0), central_frame))
            weight = F.cosine_similarity(sequence[j].unsqueeze(0), central_frame)
            weights.append(weight.item())
            

        #print("weights: ",weights)
        total_sum = sum(weights)
        #print("sum weights: ",total_sum)

        for z in range(num_frames):
            normalized_weights[z] = weights[z]/total_sum
        
       # print("normalized_weights: ",normalized_weights)
        
        for i in range(num_frames):
           # print("cosine sim score: ", F.cosine_similarity(sequence[j].unsqueeze(0), central_frame))
            weighted_seq.append(sequence[i] * weights[i])

      #  print(weighted_seq)
        # multiply each frame in the sequence by its weight

        return torch.stack(weighted_seq)


# Attention Layer Seqnet
class al_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(al_seqNet, self).__init__()
        self.inDims = inDims
        self.self_attention_layer = SelfAttentionLayer(4096)
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # print("x shape for AL: ",x.shape)

        al_x = x[0]
        #print("cosine_sim_x shape after slice: ",cosine_sim_x.shape)
        al_x = self.self_attention_layer(al_x)
        x = al_x.unsqueeze(0).expand_as(x).clone() # shape: [24,10,4096] clone to avoid gradient computing error
        x = x.permute(0,2,1) # shape: [24,4096,10]
        x = torch.sum(x, -1)
        # print("x_al shape",x.shape)

        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, sequence):
        query = self.query_linear(sequence)
        key = self.key_linear(sequence)
        value = self.value_linear(sequence)
        
#         print("before attention_scores: ",tensor_memory_usage_in_MB(sequence))
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
#         print("attention_scores: ",tensor_memory_usage_in_MB(attention_scores))
        attention_weights = torch.softmax(attention_scores, dim=-1) #potentially not use softmax maybe we use something like first order norm instead
#         print("attention weights: ",tensor_memory_usage_in_MB(attention_weights))
        attended_sequence = torch.matmul(attention_weights, value)
#         print("attended_sequence: ",tensor_memory_usage_in_MB(attended_sequence))
        # print("attended_sequence shape: ",attended_sequence.shape)
        return attended_sequence

    
class Delta(nn.Module):
    def __init__(self, inDims, seqL):

        super(Delta, self).__init__()
        self.inDims = inDims
        self.weight = (np.ones(seqL,np.float32))/(seqL/2.0)
        self.weight[:seqL//2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(self.weight),requires_grad=False)

    def forward(self, x):
        # make desc dim as C
        x = x.permute(0,2,1) # makes [B,T,C] as [B,C,T]
        delta = torch.matmul(x,self.weight)

        return delta


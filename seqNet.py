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
        print("x shape: ",x.shape)
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1)
        print("sequence feature shape",seqFt.shape)
        return seqFt
        

class cosine_seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(cosine_seqNet, self).__init__()
        self.inDims = inDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        # print(type(x))
        # print("shape for cosine before squeeze: ", x.shape)
        # print("shape for cosine: ", x.shape) #[24, 5, 4096]
        cosine_sim_x = x[0]
      #  print("x[0] before, ",x[0])
        #print("cosine_sim_x shape after slice: ",cosine_sim_x.shape)
        cosine_sim_x = self.cosine_sim(cosine_sim_x)
        x[0] = cosine_sim_x
        # print("x[0] after, ",x[0])
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
    
        # print("x after cosine and permute",x.shape)
        # print("x input shape: ", x.shape)
        # print("x before permute: ",x.shape)
        x = x.permute(0,2,1) 
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
        # print("central_frame shape: ", central_frame.shape)
        # print("central idx: ", (num_frames // 2) + 1)

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


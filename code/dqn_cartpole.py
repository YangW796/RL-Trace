import collections
import random

import numpy as np
import torch.nn.functional as F
import torch

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)
        
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    
    def sample(self,batch_size):
        transitons=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitons)
        return np.array(state),action,reward,np.array(next_state),done
    
    def size(self):
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.action_dim=action_dim
        self.q_net=Qnet(state_dim,hidden_dim,action_dim).to(device)   
        self.optimizer=torch.nn.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update 
        self.count=0
        self.device=device
    
    def take_action(self,atate):
        if np.random().random()< self.epsilon:
            action=np.random.randint(self.action_dim)
        else :
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            self.q_net(state).argmax().item()
    
    def update(self, transition_dict):
        
        return
    

            
            
            
        
        
         
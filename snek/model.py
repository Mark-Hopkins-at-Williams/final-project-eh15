import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    '''
    1. Applies first linear layer (self.linear1) to the input tensor x.
    2. Applies activation function to the output of the first linear layer.
    3. Applies second linear layer (self.linear2) to the output of the activation function.
    4. Returns the final output tensor.
    '''
    def forward(self, x):
        #x = F.relu(self.linear1(x)) #also good performer
        #x = F.gelu(self.linear1(x))
        x = F.leaky_relu(self.linear1(x)) #best performer
        #x = torch.tanh(self.linear1(x))
        #x = F.sigmoid(self.linear1(x))
        #x = F.softmax(self.linear1(x), dim = 0)
    
        #x = F.elu(self.linear1(x))

        x = self.linear2(x)
        return x

    #saves the model's state dictionary
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.mse = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimization algo

    def train_step(self, state, action, reward, next_state, gameover):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        #unsqueeze to match dim
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameover = (gameover, )

        # 1: predicted Q values with current state
        predQ = self.model(state)

        targetQ = predQ.clone()
        for idx in range(len(gameover)):
            Q_new = reward[idx]
            if not gameover[idx]:
                # Bellman equation:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            targetQ[idx][torch.argmax(action[idx]).item()] = Q_new

        # Apply the loss function:
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if game is not over
        self.optimizer.zero_grad()
        loss = self.mse(targetQ, predQ)
        loss.backward() #back propagation to compute gradients

        self.optimizer.step() #update model parameters based on the computed gradients




import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

btc = pd.read_csv('btc.csv', parse_dates=['Date'])

def create_dataset(bitcoin_data_df, lookback_size=5):
    bitcoin_data_df = bitcoin_data_df.loc[:, bitcoin_data_df.columns != 'Symbol']
    bitcoin_data_df.set_index(pd.DatetimeIndex(bitcoin_data_df['Date']), inplace=True)
    start_date = bitcoin_data_df.index.min()
    print(f"start date:{start_date}")
    bitcoin_data_df['days'] = (bitcoin_data_df.Date - start_date).transform(lambda x: x.days)
    bitcoin_data_df.sort_values('days', inplace=True)

    plt.plot(bitcoin_data_df['Date'], bitcoin_data_df['Close'])
    plt.savefig('bitcoin_trend.png')
    bitcoin_data_df.drop(['Date'], axis=1, inplace=True)
    len_traindataset = (int) (len(bitcoin_data_df) * 0.9)
    train_dataset = bitcoin_data_df.iloc[:len_traindataset]
    test_dataset = bitcoin_data_df.iloc[len_traindataset:]
    print(train_dataset.head())

    train_x_raw = train_dataset.loc[:, train_dataset.columns != 'Close'].values
    train_y_raw = train_dataset.loc[:, train_dataset.columns == 'Close'].values
    test_x_raw = test_dataset.loc[:, test_dataset.columns != 'Close'].values
    test_y_raw = test_dataset.loc[:, test_dataset.columns == 'Close'].values
    print('train duration:', len(train_dataset), train_dataset.days.iloc[0],train_dataset.days.iloc[-1], train_dataset.index.min(), train_dataset.index.max())
    print('test duration:', len(test_dataset), test_dataset.days.iloc[0], test_dataset.days.iloc[-1], test_dataset.index.min(), test_dataset.index.max())

    x_scaler = MinMaxScaler(feature_range=(0,1))
    x_scaler.fit(train_x_raw)
    train_x_raw = x_scaler.transform(train_x_raw)
    test_x_raw = x_scaler.transform(test_x_raw)

    y_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler.fit(train_y_raw)
    train_y_raw = y_scaler.transform(train_y_raw)
    test_y_raw = y_scaler.transform(test_y_raw)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(len(train_dataset) - lookback_size-1):
        train_X.append(torch.tensor(train_x_raw[i:i+lookback_size], dtype=torch.float))
        train_Y.append(torch.tensor(train_y_raw[i+lookback_size], dtype=torch.float))

    for i in range(len(test_dataset) - lookback_size-1):
        test_X.append(torch.tensor(test_x_raw[i:i+lookback_size], dtype=torch.float))
        test_Y.append(torch.tensor(test_y_raw[i+lookback_size], dtype=torch.float))

    train_data = list(zip(train_X, train_Y))
    test_data = list(zip(test_X, test_Y))
    return train_data, test_data

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        output, _ = self.lstm(x.view(-1, 1, self.input_size))
        return self.linear(output[-1].view(-1, self.hidden_size))

def train(train_data, model, optimizer, loss_criterion, epoch):
    model.train()
    training_loss = 0
    for idx, sample in enumerate(train_data):
        optimizer.zero_grad()
        input_X, target = sample
        input_X, target = input_X.to(device), target.to(device)
        output = model(input_X)
        loss = loss_criterion(output, target)
        training_loss += loss
        loss.backward()
        optimizer.step()
    training_loss /= len(train_data)
    if epoch % 10 == 0:
        print(f"Training loss at epoch:{epoch} is:{training_loss}")

def validate(test_data, model, loss_criterion, epoch):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for idx, sample in enumerate(test_data):
            input_X, target = sample
            input_X, target = input_X.to(device), target.to(device)
            output = model(input_X)
            loss = loss_criterion(output, target)
            validation_loss += loss
        validation_loss /= len(test_data)
        if epoch % 10 == 0:
            print(f"Validation loss at epoch:{epoch} is:{validation_loss}")

def plot_predictions(model, train_dataset, test_dataset):
    train = []
    train_pred = []
    test = []
    test_pred = []
    plt.clf()
    for x,target in train_dataset:
        x = x.to(device)
        pred = model(x)
        train_pred.append(pred.item())
        train.append(target.item())
    
    for x,target in test_dataset:
        x = x.to(device)
        pred = model(x)
        test_pred.append(pred.item())
        test.append(target.item())

    train_days = np.arange(len(train_dataset))
    test_days = np.arange(len(train_dataset), len(train_dataset) + len(test_dataset), 1)
    plt.plot(train_days, train)
    plt.plot(test_days, test)
    plt.plot(train_days, train_pred)
    plt.plot(test_days, test_pred)
    plt.legend(['train', 'test', 'train_pred', 'test_pred'])
    plt.savefig('model_pred_comparison.png')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    bitcoin_train_dataset, bitcoin_test_dataset = create_dataset(btc)
    train_dataloader = DataLoader(bitcoin_train_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(bitcoin_test_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = Model(6, 32, 1)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = F.mse_loss
    for epoch in range(50):
        train(train_dataloader, model, optimizer, criterion, epoch)
        validate(test_dataloader, model, criterion, epoch)
    plot_predictions(model, bitcoin_train_dataset, bitcoin_test_dataset)

# # \begin{array}{ll} \\
# #             i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
# #             f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
# #             g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
# #             o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
# #             c_t = f_t c_{(t-1)} + i_t g_t \\
# #             h_t = o_t \tanh(c_t) \\
# #         \end{array}
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, layers=1):
#         ## forget gate
#         self.W_if = nn.Linear(input_size, hidden_size, bias=True)
#         self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)
#         ## sigmoid input gate
#         self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
#         self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)
#         ## tanh input gate
#         self.W_gi = nn.Linear(input_size, hidden_size, bias=True)
#         self.W_gh = nn.Linear(hidden_size, hidden_size, bias=True)
#         ## output gate
#         self.W_oi = nn.Linear(input_size, hidden_size, bias=True)
#         self.W_oh = nn.Linear(input_size, hidden_size, bias=True)
#         self.
        
#     def forward(self, h_t, x):
#         while(layers > 0):
#             forget_part = F.sigmoid(self.W_if + self.W_hf)
#             sigmoid_input_part = F.sigmoid(self.W_ii + self.W_hi)
#             tan_input_part = F.tanh(self.W_gi + self.W_gh)
#             C_t = sigmoid_input_part * tan_input_part
#             if C_t_1 is not None:
#                 C_t = C_t_1 * forget_part + C_t
#             output_part = F.sigmoid(self.W_oi + self.W_oh)
#             h_t = F.tanh(C_t) * output_part
#         return C_t, h_t



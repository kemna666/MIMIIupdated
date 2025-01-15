from net.CNN import CNN
from feeder.feeder import MIMIIDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from util.parser import readconfig
import time
import yaml
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt





class Process():
    def __init__(self,args):
        self.args=args
        self.train_accuracy = []
        self.epoch_data = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.load_Crition()
        self.train()
        self.accuracy()
        self.plot()
    def load_data(self):
        if self.args.dataset == 'MIMII':
            self.dataset=MIMIIDataset(self.args.pkl_file_path)
        else:
            raise ValueError("数据集不对！")
        self.train_data, self.test_data = train_test_split(self.dataset, test_size=self.args.test_size, random_state=self.args.random_state)
        self.train_loader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)
    def load_model(self):
        if self.args.model == 'CNN':
            self.model = CNN(input_dim=13,hidden_dim=128,output_dim=13,length=153).to(self.device)
        else:
            raise ValueError("请选择正确的模型！")
    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.arg.lr,momentum=0.9)
        if self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        else:
            raise ValueError("请选择正确的优化器！")
    def load_Crition(self):
        if self.args.loss == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("请选择正确的损失函数！")
    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        return Batch.from_data_list(batch)

    def train(self):
        for self.epoch in range(self.args.epochs):
            self.model.train()
            train_loss = 0
            for data in self.train_loader:
                device_index = data.y.to(self.device)  # 输入设备索引
                mfcc_features = data.x.to(self.device)  # 输入MFCC特征
                output = self.model(mfcc_features, device_index)  # 输出标签
                train_loss = self.loss(output, device_index)
                
                train_loss.backward()
                self.optimizer.step()
                print(f'{self.epoch+1} loss={train_loss.item()}\n')
            self.accuracy()
    def accuracy(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in self.test_loader:
                device_index = data.y.to(self.device)  # 输入设备索引
                mfcc_features = data.x.to(self.device)  # 输入MFCC特征
                output = self.model(mfcc_features, device_index)  # 输出标签
                _, predicted = torch.max(output.data, 1)
                total += device_index.size(0)
                correct += (predicted == device_index).sum().item()
            acc = correct / total
        print(f'Accuracy: {acc:.4f}')
        self.epoch_data.append(self.epoch)
        self.train_accuracy.append(acc)
    def plot(self):
        xpoint = self.epoch_data
        ypoint = self.train_accuracy
        plt.plot(xpoint, ypoint)
        plt.savefig(f'./output/accuracy_date{time.time()}.png')
        plt.show()


if __name__=='__main__':
    
    config_reader = readconfig()
    Process(config_reader)
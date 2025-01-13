from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.data import Batch
import torch.nn as nn 
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#定义模型
from torch.utils.data import Dataset
import pickle
class MIMIIDataset(Dataset):    
    def __init__(self, pkl_file_path):
        # 加载.pkl文件中的数据
        with open(pkl_file_path, 'rb') as file:
            self.data = pickle.load(file)
        self.merged_data =[]
        for snr_data in self.data:
            self.merged_data.extend(snr_data)
    
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        # 根据索引idx返回一个样本的特征和标签
        mfcc_features = self.merged_data[idx][0]
        device_index = self.merged_data[idx][1]
        label_index =   self.merged_data[idx][2]
       # 将MFCC特征转换为Tensor
        mfcc_features = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        return Data(x=mfcc_features, edge_index=label_index, y=device_index)


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, length):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=1, padding=1)
        self.Conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        conv1_out_length = (length + 2*1 - 7) // 1 + 1
        pool1_out_length = (conv1_out_length - 3) // 2 + 1
        conv2_out_length = (pool1_out_length + 2*1 - 3) // 1 + 1
        pool2_out_length = (conv2_out_length - 3) // 2 + 1
        self.fc = nn.Linear(output_dim * pool2_out_length, 153)  # 输出维度为153

    def forward(self, x, device_index):
        # 将device_index转换为one-hot编码
        device_one_hot = torch.nn.functional.one_hot(device_index, num_classes=self.output_dim)
        device_one_hot = device_one_hot.float().unsqueeze(2)  # 添加一个维度以匹配卷积层的输入

        # 将device_one_hot与MFCC特征拼接
        x = torch.cat((x, device_one_hot), dim=2)

        x = self.Conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.Conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # 只返回标签
    

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    return Batch.from_data_list(batch)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN(input_dim=13,hidden_dim=128,output_dim=13,length=153).to(device)
dataset = MIMIIDataset('./data/data.pkl')
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

#定义训练参数
#训练轮数
num_epochs = 20
# 开始训练
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

train_accuracy = []
epoch_data = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in train_loader:
        device_index = data.y.to(device)  # 输入设备索引
        mfcc_features = data.x.to(device)  # 输入MFCC特征
        output = model(mfcc_features, device_index)  # 输出标签
        train_loss = loss(output, device_index)
        
        train_loss.backward()
        optimizer.step()
        print(f'{epoch+1} loss={train_loss.item()}\n')
        # 一轮计算一次准确率
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            device_index = data.y.to(device)  # 输入设备索引
            mfcc_features = data.x.to(device)  # 输入MFCC特征
            output = model(mfcc_features, device_index)  # 输出标签
            _, predicted = torch.max(output.data, 1)
            total += device_index.size(0)
            correct += (predicted == device_index).sum().item()
        acc = correct / total
    print(f'Accuracy: {acc:.4f}')
    epoch_data.append(epoch)
    train_accuracy.append(acc)
xpoint = epoch_data
ypoint = train_accuracy
plt.plot(xpoint, ypoint)
plt.savefig('./data/train_accuracy.png')
plt.show()
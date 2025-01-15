import argparse
import os
import yaml

class readconfig:
    def __init__(self):

        self.parser=self.getparserfromyaml()
        self.args=None
        self.process_parser()
    def getparserfromyaml(self): 
        parser = argparse.ArgumentParser(description='MIMII参数')
        parser.add_argument('--config',type=str,required=True,help='配置文件路径')
        parser.add_argument('--pkl_file_path',type=str,required=True,help='pkl文件路径')
        parser.add_argument('--model',type=str,required=True,help='模型')
        parser.add_argument('--batch_size',type=int,required=True,help='批量大小')
        parser.add_argument('--input_dim',type=int,required=False,help='输入维度')
        parser.add_argument('--hidden_dim',type=int,required=False,help='隐藏层维度')
        parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
        parser.add_argument('--optimizer',type=str,required=True,help='优化器')
        parser.add_argument('--lr',type=float,required=True,help='学习率')
        parser.add_argument('--loss',type=str,required=True,help='损失函数')
        return parser
    def process_parser(self):
        # 获取解析后的参数
        self.args = self.parser.parse_args()
        # 判断配置文件是否为 yaml 文件
        if self.args.config.endswith('.yaml'):
            # 打开 yaml 文件
            with open(args.config) as f:
                # 安全加载 yaml 文件内容
                self.config = yaml.load(f,Loader=yaml.FullLoader)
            # 从 yaml 文件中获取相应的参数并赋值给解析后的参数
            self.args.pkl_file_path = self.config.get('data', {}).get('pkl_file_path', self.args.pkl_file_path)
            self.args.batch_size = self.config.get('data', {}).get('batch_size', self.args.batch_size)
            self.args.input_dim = self.config.get('model', {}).get('input_dim', self.args.input_dim)
            self.args.hidden_dim = self.config.get('model', {}).get('hidden_dim', self.args.hidden_dim)
            self.args.hidden_dim = self.config.get('model', {}).get('model', self.args.model)
            self.args.epochs = self.config.get('train', {}).get('epochs', self.args.epochs)
            self.args.optimizer = self.config.get('optimizer', {}).get('optim', self.args.optimizer)
            self.args.lr = self.config.get('optimizer', {}).get('lr', self.args.lr)
            self.args.loss = self.config.get('train', {}).get('loss', self.args.loss)
            print(self.args)
        else:
            raise ValueError('请选择有效的 yaml 文件路径！')


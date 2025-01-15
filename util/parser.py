import argparse
import os
import yaml

class readconfig:
    def __init__(self):

        self.parser=self.getparserfromyaml()
        self.process_parser()
        self.pkl_file_path = self.config['data']['pkl_file_path']
        self.dataset = self.config['data']['dataset']
        self.batch_size = self.config['data']['batch_size']
        self.test_size = self.config['spilt']['testsize']
        self.random_state = self.config['spilt']['random_state']
        self.model = self.config['model']['model']
        self.input_dim = self.config['model']['input_dim']
        self.hidden_dim = self.config['model']['hidden_dim']
        self.epochs = self.config['train']['epochs']
        self.loss = self.config['train']['loss']
        self.optimizer = self.config['optimizer']['optim']
        self.lr = self.config['optimizer']['lr']
    def getparserfromyaml(self): 
        parser = argparse.ArgumentParser(description='MIMII参数')
        parser.add_argument('--config',type=str,required=True,help='配置文件路径')
        return parser
    def process_parser(self):
        # 获取解析后的参数
        self.p = self.parser.parse_args()
        # 判断配置文件是否为 yaml 文件
        if self.p.config.endswith('.yaml'):
            # 打开 yaml 文件
            with open(self.p.config) as f:
                # 安全加载 yaml 文件内容
                self.config = yaml.safe_load(f)
        else:
            raise ValueError('请选择有效的 yaml 文件路径！')


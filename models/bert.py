import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_SourceCode.tokenization import BertTokenizer
from bert_SourceCode.modeling import BertModel, BertAttention


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class Config(object):
    def __init__(self, dataDir):
        self.model_name = 'bert'  # 模型名称
        self.train_data = dataDir + '/train.txt'  # 训练集
        self.dev_data = dataDir + '/dev.txt'  # 验证集
        self.text_data = dataDir + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(dataDir + '/class.txt').readlines()]   # label类别（几分类）
        self.save_model_path = 'trainedModel/' + self.model_name + '.pkl'  # 存储训练好的模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备使用

        self.num_classes = len(self.class_list)  # 分类数
        self.pad_size = 32  # 每句话处理的长度（短了补长了切）
        self.batch_size = 128  # mini-batch的大小
        self.hidden_size = 768  # 隐层大小
        self.learning_rate = 5e-5  # 学习率
        self.num_epochs = 3  # epoch数
        self.require_improve = 1000  # 若超过1000 batch仍无提升，则提前结束

        self.bert_SourceCode_path = './bert_SourceCode'  # bert源码
        self.pretrainedModel_path = './bert_pretrainedModel'  # 下载的bert预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainedModel_path)  # BertTokenizer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 从bert源码中加载模型
        self.bert = BertModel.from_pretrained(config.pretrainedModel_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        self.weight_W = nn.Parameter(torch.rand(config.hidden_size , config.hidden_size))
        self.weight_proj = nn.Parameter(torch.rand(config.hidden_size, 1))

    def forward(self, x):
        content = x[0]
        label = x[1]
        mask = x[2]
        _, pooled = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)
        #_=[batch_size, sequence_length, hidden_size]
        #pooled=[batch_size, hidden_size]  
        #pooled is Last layer hidden-state of the first token ('CLS') of the sequence,it maybe "context vector"
        
        ####below is a kind of self attention
        u = torch.tanh(torch.matmul(_, self.weight_W))
        #u=[batch_size, sequence_length, hidden_size]
        att = torch.matmul(u, self.weight_proj)
        #att=[batch_size, sequence_length,1]
        att_score = F.softmax(att, dim=1)
        #att_score=[batch_size, sequence_length,1],which sum by {seq_len}=1
        scored_x = _ * att_score
        #scored_x=[batch_size, sequence_length,hidden_size]
        #####attention_weighted_x
        
        scored_x = torch.sum(scored_x, dim=1)
        #out = self.fc(pooled)
        out=self.fc(scored_x)
        
        return out

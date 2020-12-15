# coding: UTF-8
import time
import torch
import math
import argparse
from importlib import import_module
from loadDataset import loadDataset, cost_Time, list2iter

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a models: Bert, BertCNN, ERNIE')
parser.add_argument('--predict', type=str, required=True, help='input the predict sentence')
args = parser.parse_args()
PAD, CLS = '[PAD]', '[CLS]'


def softmax(matrix):
    m_exp = [math.exp(i) for i in matrix]
    sum_m_exp = sum(m_exp)
    norm = [round(i / sum_m_exp, 3) for i in m_exp]
    return norm


def evaluate(model, content, sentence):
    model.eval()
    with torch.no_grad():
      for text, device in content:
        outputs, att_score = model(text)
        predict = torch.max(outputs.data, 1)[1].cpu().numpy()
    pos_att = {}
    pos_word = {}
    attention_list = att_score[0].cpu().numpy().tolist()
    i = 0
    att_list = [l[0] for l in attention_list]

    att_list = att_list[1:len(sentence)+1]
    att_list = softmax(att_list)
    for i in range(0, len(sentence)):
      pos_att[i] = sentence[i]

    for i in range(0, len(sentence)):
      pos_word[i] = round(att_list[i], 4)

    print(pos_att)
    print(pos_word)
    pred_list = softmax(outputs[0])
    dict = {}
    for i in range(0, config.num_classes):
      dict[labels[i][1]] = pred_list[i]
    print(dict)


def loadData(config, sentence, pad_size=32):
    contents = []
    line = sentence.strip()
    if not line:
        return
    token = config.tokenizer.tokenize(line)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    contents.append((token_ids, int(-1), seq_len, mask))
    return contents


if __name__ == '__main__':
    # 根据执行命令获取要运行的模型，并引入对应的module，使用模型对应的配置参数
    model_name = args.model
    sentence = args.predict
    model_path = import_module('models.' + model_name)
    config = model_path.Config('dataset')

    labels = [x.split(' ') for x in config.class_list]
    start_time = time.time()
    # 使用训练好的模型
    model = model_path.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_model_path), False)
    print("Cost Time: ", cost_Time(start_time))
    # 预测
    start_time = time.time()
    contentList = loadData(config, sentence)
    contentIter = list2iter(contentList, config)
    evaluate(model, contentIter, sentence)
    print("Cost Time: ", cost_Time(start_time))



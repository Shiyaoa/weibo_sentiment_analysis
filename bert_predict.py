from importlib import import_module

import numpy as np
import torch
import time
from utils import get_time_dif, build_iterator

def test(model, sentence):
    # test
    model_path = 'THUCNews/saved_dict/bert.ckpt'
    model.load_state_dict(torch.jit.load(model_path), False)
    # model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    model.to(device)
    start_time = time.time()
    test_report = evaluate(model, sentence)
    print(test_report)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(model, sentence):
    model.eval()
    with torch.no_grad():
      for text, label in sentence:
        # text.to(device)

        outputs = model(text)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        return predic


PAD, CLS = '[PAD]', '[CLS]'


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
    contents.append((token_ids, -1, seq_len, mask))
    print(contents)
    return contents



def predict(str):
    x = import_module('models.bert')
    config = x.Config('THUCNews')
    device = torch.device('cuda:0')
    model = x.Model(config).to(config.device)
    t = loadData(config, str)
    test_data = build_iterator(t, config)
    test(model, test_data)

predict("不过，喜欢4.0原生态的界面！")
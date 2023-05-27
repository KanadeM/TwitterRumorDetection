import json
from torch.utils.data import Dataset
import os
import time
import torch

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

class CommonDataSet(Dataset):
  def __init__(self, tokenizer, max_length, data_dir,
              data_file_name, label_file_name, n_obs=None):
    super().__init__()
    self.data = []
    self.label = [] 
    for ids, label in zip(open(data_file_name).readlines(), open(label_file_name).readlines()):
      tmp_ids_lst = ids.strip().split(",")
      tweet_json_lst = []
      if not os.path.exists(data_dir + tmp_ids_lst[0] + "json"):
        continue                
      for i in tmp_ids_lst:
        tweet_path = data_dir + i + ".json"
        if os.path.exists(tweet_path):
          tweet_json_lst.append(json.load(open(tweet_path, "r")))
          tweet_json_lst = sorted(tweet_json_lst, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
          self.data.append(tweet_json_lst)
          self.label.append(0 if label.strip() == "nonrumour" else 1)
    self.max_length = max_length
    self.tokenizer = tokenizer
    self.n_obs = n_obs
  def __len__(self):
    if self.n_obs is None:
      return len(self.label)
    else:
      return self.n_obs
  def __getitem__(self, index):
    return self.data[index], self.label[index]
  def collate_fn(self, batch):
    input_text = []
    labels = []
    for x, label in batch:
      x_text = []
      for y in x:
          x_text.append(preprocess(y["text"]))
      input_text.append(self.tokenizer.sep_token.join(x_text))
      labels.append(label)

    src_text = self.tokenizer(
      input_text,
      max_length=self.max_length,
      padding=True,
      return_tensors="pt",
      truncation=True,
    )

    batch_encoding = dict()
    batch_encoding["input_text"] = src_text.input_ids
    batch_encoding["attn_text"] = src_text.attention_mask
    batch_encoding["label"] = torch.LongTensor(labels)

    return batch_encoding


class TestDataSet(CommonDataSet):
  def __init__(self, tokenizer, max_length, data_dir,
              data_file_name, n_obs=None):
    super().__init__()
    self.data = []
    for ids, label in open(data_file_name).readlines():
      tmp_ids_lst = ids.strip().split(",")
      tweet_json_lst = []
      if not os.path.exists(data_dir + tmp_ids_lst[0] + "json"):
        continue                
      for i in tmp_ids_lst:
        tweet_path = data_dir + i + ".json"
        if os.path.exists(tweet_path):
          tweet_json_lst.append(json.load(open(tweet_path, "r")))
          tweet_json_lst = sorted(tweet_json_lst, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
          self.data.append(tweet_json_lst)
    self.max_length = max_length
    self.tokenizer = tokenizer
    self.n_obs = n_obs

  def collate_fn(self, batch):
    input_text = []
    for x in batch:
        x_text = []
        for y in x:
            x_text.append(preprocess(y["text"]))
        input_text.append(self.tokenizer.sep_token.join(x_text))

    src_text = self.tokenizer(
        input_text,
        max_length=self.max_length,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )

    batch_encoding = dict()
    batch_encoding["input_text"] = src_text.input_ids
    batch_encoding["attn_text"] = src_text.attention_mask

    return batch_encoding


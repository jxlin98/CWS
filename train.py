import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from data import read_data
from model import CWS
from utils import arg_parse, collate_fn


def train():
    args = arg_parse()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    train_dev_features = read_data('./datasets/training.txt', tokenizer)
    train_features = train_dev_features[:int(0.9 * len(train_dev_features))]
    dev_features = train_dev_features[int(0.9 * len(train_dev_features)):]

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    dev_dataloader = DataLoader(dev_features, batch_size=args.dev_batch_size, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CWS(model, args.class_size)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    total_steps = len(train_dataloader) * args.num_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print("training")

    best = -1
    start_epoch = 0
    if os.path.exists(args.path_checkpoint):
        checkpoint = torch.load(args.path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dic"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.num_epoch):
        for i, data in enumerate(train_dataloader):
            model.train()
            input_ids, input_mask, labels = data
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            loss, _ = model(input_ids, input_mask, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dic": optimizer.state_dict(),
                      "epoch": epoch}
        torch.save(checkpoint, args.path_checkpoint)

        predict, label = [], []
        for data in dev_dataloader:
            model.eval()
            input_ids, input_mask, labels = data
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                _, tag_seq = model(input_ids, input_mask, labels)

            logits = tag_seq.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i in range(len(input_mask)):
                for j, mask in enumerate(input_mask[i]):
                    if mask:
                        predict.append(logits[i][j])
                        label.append(labels[i][j])

        report = classification_report(label, predict, digits=6)
        F = float(report.split()[-2])
        print("epoch: {}    F: {}".format(epoch, F))
        print(report)

        if F > best:
            best = F
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dic": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(checkpoint, args.best_checkpoint)


if __name__ == '__main__':
    train()

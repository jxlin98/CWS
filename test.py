import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from sklearn.metrics import classification_report

from data import read_data
from model import CWS
from utils import arg_parse, collate_fn, label_to_number


def test():
    args = arg_parse()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    test_features = read_data('./datasets/test.txt', tokenizer)
    test_dataloader = DataLoader(test_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CWS(model, args.class_size)
    model.to(device)

    checkpoint = torch.load(args.best_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    f = open('./result/result_bert.txt', 'w')
    f1 = open('./datasets/test.txt', 'r')
    sentences = f1.readlines()

    predicts, labelss = [], []
    for data in test_dataloader:
        model.eval()
        input_ids, input_mask, labels = data
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            _, tag_seq = model(input_ids, input_mask, labels)

        logits = tag_seq.to('cpu').numpy()

        input_ids = input_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        labels = labels.to('cpu').numpy()

        for i in range(len(input_mask)):
            predict, label = [], []
            for j, mask in enumerate(input_mask[i]):
                if mask:
                    predict.append(logits[i][j])
                    label.append(labels[i][j])
            predicts.append(predict)
            labelss.append(label)

    for i in range(len(sentences)):
        sentence = list("".join(sentences[i].split()))
        word = []
        for j in range(len(sentence)):
            if predicts[i][j + 1] == label_to_number['S']:
                f.write(sentence[j] + "  ")
            elif predicts[i][j + 1] == label_to_number['B']:
                if word:
                    word = "".join(word)
                    f.write(word + "  ")
                    word = []
                    word.append(sentence[j])
                else:
                    word.append(sentence[j])
            elif predicts[i][j + 1] == label_to_number['I']:
                word.append(sentence[j])
            elif predicts[i][j + 1] == label_to_number['E']:
                word.append(sentence[j])
                word = "".join(word)
                f.write(word + "  ")
                word = []
            j += 1
        f.write("\n")

    f.close()
    f1.close()
    report = classification_report(sum(labelss, []), sum(predicts, []), digits=6)
    print(report)


if __name__ == '__main__':
    test()

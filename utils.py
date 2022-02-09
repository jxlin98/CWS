import argparse
import numpy as np

import torch
import torch.nn.functional as F

label_to_number = {'O': 0, 'S': 1, 'B': 2, 'I': 3, 'E': 4, 'CLS': 5, 'SEP': 6}

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--dev_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)

    parser.add_argument("--class_size", default=7, type=int)

    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num-epoch", default=30, type=int)
    parser.add_argument("--path_checkpoint", default="./checkpoint/model.pkl", type=str)
    parser.add_argument("--best_checkpoint", default="./checkpoint/best.pkl", type=str)
    args = parser.parse_args()

    return args


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [label_to_number['O']] * (max_len - len(f["labels"])) for f in batch] # padding

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    output = (input_ids, input_mask, labels)
    return output


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention

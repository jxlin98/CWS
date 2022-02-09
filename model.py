import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import process_long_input


class CWS(nn.Module):
    def __init__(self, model, class_size):
        super(CWS, self).__init__()
        self.bert = model
        self.config = model.config
        self.class_size = class_size

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.matrix = nn.Linear(self.config.hidden_size, self.class_size)

    def forward(self, input_ids, input_mask, labels):
        sequence_output, _ = process_long_input(self.bert, input_ids, input_mask, [self.config.cls_token_id], [self.config.sep_token_id])
        sequence_output = self.dropout(sequence_output)
        logits = self.matrix(sequence_output)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        total_loss = loss_fct(logits.view(-1, self.class_size), labels.view(-1))
        tag_seq = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        return total_loss, tag_seq
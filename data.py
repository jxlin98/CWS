from transformers import BertTokenizer

from utils import label_to_number


def read_data(path, tokenizer):
    with open(path, 'r') as f:
        lines = f.readlines()

    features = []

    for line in lines:
        labels = []
        labels.append(label_to_number['CLS'])
        words = line.split()
        for word in words:
            if len(word) == 1:
                labels.append(label_to_number['S'])
            else:
                labels.append(label_to_number['B'])
                for i in range(1, len(word) - 1):
                    labels.append(label_to_number['I'])
                labels.append(label_to_number['E'])
        labels.append(label_to_number['SEP'])

        sentence = []
        for l in list("".join(words)):
            sentence.append(l)
        input_ids = tokenizer.convert_tokens_to_ids(sentence)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = {'input_ids': input_ids, 'labels': labels}
        features.append(feature)

    return features


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
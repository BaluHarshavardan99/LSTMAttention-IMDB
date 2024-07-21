import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(clean_text(text))

def load_data(batch_size, device):
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    # Load datasets
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(clean_text(x)))

    def label_pipeline(x):
        return 1 if x == 'pos' else 0

    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            cleaned_text = clean_text(_text)
            processed_text = torch.tensor(text_pipeline(cleaned_text), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(len(processed_text))
            label_list.append(label_pipeline(_label))
        
        lengths, text_list, label_list = zip(*sorted(zip(lengths, text_list, label_list), key=lambda x: x[0], reverse=True))
        
        label_list = torch.tensor(label_list, dtype=torch.float32)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        
        return text_list.to(device), label_list.to(device), lengths

    train_iter, test_iter = IMDB(split=('train', 'test'))

    # Split train_iter into train and validation datasets
    train_list = list(train_iter)
    train_size = int(0.8 * len(train_list))
    valid_size = len(train_list) - train_size
    train_dataset, valid_dataset = random_split(train_list, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return vocab, train_dataloader, valid_dataloader, test_dataloader

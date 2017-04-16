#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.autograd import Variable
import tqdm

import models
from models import CharRNN
from utils import variable, cuda


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus')
    arg('root', help='checkpoint root')
    arg('--mode', choices=['train', 'validate'], default='train')
    arg('--model', default='CharGRU')
    arg('--batch-size', type=int, default=4)
    arg('--window-size', type=int, default=256)
    arg('--hidden-size', type=int, default=128)
    arg('--n-layers', type=int, default=1)
    arg('--lr', type=float, default=0.01)
    arg('--n-epochs', type=int, default=10)
    arg('--epoch-batches', type=int,
        help='force epoch to have given number of batches')
    arg('--valid-corpus', help='path to validation corpus')
    arg('--valid-batches', type=int, help='validate on first N batches')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True)
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True))

    corpus = Path(args.corpus).read_text(encoding='utf8')
    vocab_file = root.joinpath('vocab.json')
    if vocab_file.exists():
        char_to_id = json.loads(vocab_file.read_text(encoding='utf8'))
    else:
        char_to_id = get_char_to_id(corpus)
        with vocab_file.open('wt', encoding='utf8') as f:
            json.dump(char_to_id, f, ensure_ascii=False, indent=True)

    n_characters = len(char_to_id)
    model = getattr(models, args.model)(
        input_size=n_characters,
        hidden_size=args.hidden_size,
        output_size=n_characters,
        n_layers=args.n_layers,
    )  # type: CharRNN
    model_file = root.joinpath('model.pt')
    if model_file.exists():
        state = torch.load(str(model_file))
        model.load_state_dict(state['state'])
        epoch = state['epoch']
    else:
        epoch = 1
    model = cuda(model)
    criterion = torch.nn.CrossEntropyLoss()

    if args.mode == 'train':
        train(args, model, epoch, corpus, char_to_id, criterion, model_file)
    elif args.mode == 'validate':
        if not args.valid_corpus:
            parser.error(
                'Pass path to validation corpus via --valid-corpus')
        validate(args, model, criterion, char_to_id)
    else:
        parser.error('Unexpected mode {}'.format(args.mode))


def train(args, model: CharRNN,
          epoch, corpus, char_to_id, criterion, model_file):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_chars = args.window_size * args.batch_size
    save = lambda ep: torch.save({
        'state': model.state_dict(), 'epoch': ep}, str(model_file))
    for epoch in range(epoch, args.n_epochs + 1):
        try:
            losses = []
            n_iter = args.epoch_batches or (len(corpus) // batch_chars)
            tr = tqdm.tqdm(total=n_iter * batch_chars)
            tr.set_description('Epoch {}'.format(epoch))
            model.train()
            for _ in range(n_iter):
                inputs, targets = random_batch(
                    corpus,
                    batch_size=args.batch_size,
                    window_size=args.window_size,
                    char_to_id=char_to_id,
                )
                loss = train_model(
                    model, criterion, optimizer, inputs, targets)
                losses.append(loss)
                tr.update(batch_chars)
                tr.set_postfix(loss=np.mean(losses[-100:]))
            save(ep=epoch + 1)
        except KeyboardInterrupt:
            print('\nGot Ctrl+C, saving checkpoint...')
            save(ep=epoch)
            print('done.')
            return
        if args.valid_corpus:
            validate(args, model, criterion, char_to_id)
    print('Done training for {} epochs'.format(args.n_epochs))


def validate(args, model: CharRNN, criterion, char_to_id):
    model.eval()
    valid_corpus = Path(args.valid_corpus).read_text(encoding='utf8')
    batch_size = 1
    window_size = args.window_size
    hidden = cuda(model.init_hidden(batch_size))
    loss = n = 0
    n_iter = ((window_size * args.valid_batches) if args.valid_batches
              else len(valid_corpus))
    for idx in range(0, min(n_iter, len(valid_corpus) - 1), window_size):
        chunk = valid_corpus[idx: idx + window_size + 1]
        inputs = variable(char_tensor(chunk[:-1], char_to_id).unsqueeze(0))
        targets = variable(char_tensor(chunk[1:], char_to_id).unsqueeze(0))
        for c in range(inputs.size(1)):
            output, hidden = model(inputs[:, c], hidden)
            loss += criterion(output.view(batch_size, -1), targets[:, c])
            n += 1
    mean_loss = loss.data[0] / n
    print('Validation loss: {:.3}'.format(mean_loss))


def train_model(model: CharRNN, criterion, optimizer,
                inputs: Variable, targets: Variable) -> float:
    batch_size = inputs.size(0)
    window_size = inputs.size(1)
    hidden = cuda(model.init_hidden(batch_size))
    model.zero_grad()
    loss = 0
    for c in range(window_size):
        output, hidden = model(inputs[:, c], hidden)
        loss += criterion(output.view(batch_size, -1), targets[:, c])
    loss.backward()
    optimizer.step()
    return loss.data[0] / window_size


CharToId = Dict[str, int]


def random_batch(corpus: str, *, batch_size: int, window_size: int,
                 char_to_id: CharToId) -> Tuple[Variable, Variable]:
    inputs = torch.LongTensor(batch_size, window_size)
    targets = torch.LongTensor(batch_size, window_size)
    for bi in range(batch_size):
        idx = random.randint(0, len(corpus) - window_size)
        chunk = corpus[idx: idx + window_size + 1]
        inputs[bi] = char_tensor(chunk[:-1], char_to_id)
        targets[bi] = char_tensor(chunk[1:], char_to_id)
    return variable(inputs), variable(targets)


UNK = '<UNK>'


def get_char_to_id(corpus: str, max_chars=1024) -> CharToId:
    counts = Counter(corpus)
    char_to_id = {
        c: i for i, (c, _) in enumerate(counts.most_common(max_chars - 1))}
    char_to_id[UNK] = len(char_to_id)
    return char_to_id


def char_tensor(string: str, char_to_id: CharToId) -> torch.LongTensor:
    tensor = torch.LongTensor(len(string))
    for i, c in enumerate(string):
        try:
            tensor[i] = char_to_id[c]
        except KeyError:
            tensor[i] = char_to_id[UNK]
    return tensor


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.autograd import Variable
import tqdm

from models import CharRNN
from utils import variable, cuda


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('corpus', type=Path)
    # arg('root', type=Path)
    arg('--batch-size', type=int, default=4)
    arg('--window-size', type=int, default=256)
    arg('--n-epochs', type=int, default=10)
    arg('--hidden-size', type=int, default=128)
    arg('--n-layers', type=int, default=1)
    arg('--lr', type=float, default=0.01)
    args = parser.parse_args()

    with args.corpus.open(encoding='utf8') as f:
        corpus = f.read()

    char_to_id = get_char_to_id(corpus)
    n_characters = len(char_to_id)
    model = CharRNN(
        input_size=n_characters,
        hidden_size=args.hidden_size,
        output_size=n_characters,
        n_layers=args.n_layers,
    )
    model = cuda(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    batch_chars = args.window_size * args.batch_size
    for epoch in range(args.n_epochs):
        losses = []
        tr = tqdm.tqdm(total=len(corpus))
        tr.set_description('Epoch {}'.format(epoch + 1))
        for _ in range(len(corpus) // batch_chars):
            inputs, targets = random_batch(
                corpus,
                batch_size=args.batch_size,
                window_size=args.window_size,
                char_to_id=char_to_id,
            )
            loss = update_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
            )
            losses.append(loss)
            tr.update(batch_chars)
            tr.set_postfix(loss=np.mean(losses[-100:]))


def update_model(*, model: CharRNN, criterion, optimizer,
                 inputs: Variable, targets: Variable
                 ) -> float:
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
        start_index = random.randint(0, len(corpus) - window_size)
        end_index = start_index + window_size + 1
        chunk = corpus[start_index:end_index]
        inputs[bi] = char_tensor(chunk[:-1], char_to_id)
        targets[bi] = char_tensor(chunk[1:], char_to_id)
    return variable(inputs), variable(targets)


def get_char_to_id(corpus: str, max_chars=1024) -> CharToId:
    counts = Counter(corpus)
    return {c: i for i, (c, _) in enumerate(counts.most_common(max_chars))}


def char_tensor(string: str, char_to_id: CharToId) -> torch.LongTensor:
    tensor = torch.LongTensor(len(string))
    for i, c in enumerate(string):
        tensor[i] = char_to_id[c]
    return tensor


if __name__ == '__main__':
    main()

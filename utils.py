import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch.cuda
from torch.autograd import Variable


cuda_is_available = torch.cuda.is_available()


def variable(x):
    return cuda(Variable(x))


def cuda(x):
    return x.cuda() if cuda_is_available else x


def write_event(log, **data):
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def plot(*args, params=False):
    """ Use in the notebook like this:
    plot('./runs/oc2', './runs/oc1', 'loss', 'valid_loss')
    """
    paths, keys = [], []
    for x in args:
        if x.startswith('.') or x.startswith('/'):
            paths.append(x)
        else:
            keys.append(x)
    plt.figure(figsize=(12, 8))
    for path in sorted(paths):
        path = Path(path)
        with path.joinpath('train.log').open() as f:
            events = [json.loads(line) for line in f]
        if params:
            print(path)
            pprint(json.loads(path.joinpath('params.json').read_text()))
        for key in sorted(keys):
            xs, ys = [], []
            for i, e in enumerate(events):
                if key in e:
                    xs.append(i)
                    ys.append(e[key])
            if xs:
                plt.plot(xs, ys, label='{}: {}'.format(path, key))
    plt.legend()

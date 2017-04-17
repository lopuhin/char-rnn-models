import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch.cuda
from torch.autograd import Variable


cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def plot(*args, ymin=None, ymax=None, xmin=None, xmax=None, params=False):
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

    ylim_kw = {}
    if ymin is not None:
        ylim_kw['ymin'] = ymin
    if ymax is not None:
        ylim_kw['ymax'] = ymax
    if ylim_kw:
        plt.ylim(**ylim_kw)

    xlim_kw = {}
    if xmin is not None:
        xlim_kw['xmin'] = xmin
    if xmax is not None:
        xlim_kw['xmax'] = xmax
    if xlim_kw:
        plt.xlim(**xlim_kw)

    for path in sorted(paths):
        path = Path(path)
        with path.joinpath('train.log').open() as f:
            events = [json.loads(line) for line in f]
        if params:
            print(path)
            pprint(json.loads(path.joinpath('params.json').read_text()))
        for key in sorted(keys):
            xs, ys = [], []
            for e in events:
                if key in e:
                    xs.append(e['step'])
                    ys.append(e[key])
            if xs:
                plt.plot(xs, ys, label='{}: {}'.format(path, key))
    plt.legend()

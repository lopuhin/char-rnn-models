import json
from datetime import datetime

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

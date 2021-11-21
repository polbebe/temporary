from train_models import prep_data, shuffle_batch
import torch

def parallelize(inputs, P, Q):
    busy = [False for _ in range(len(P))]
    outputs = []
    idx = 0
    while idx < len(inputs):
        for i in range(len(busy)):
            # if any process is freed up add a new input there
            if busy[i] == False:
                Q[i][0].put(inputs[idx])
                busy[i] = True
                idx += 1
                if idx >= len(inputs): break
            # check at each step to see if a process is free
            # except and do nothing otherwise
            else:
                try:
                    v = Q[i][1].get_nowait()
                    outputs.append(v)
                    busy[i] = False
                except: pass

    # collect the stragglers
    for i in range(len(busy)):
        if busy[i] == True:
            v = Q[i][1].get()
            outputs.append(v)
            busy[i] = False
    assert len(outputs) == len(inputs)
    return outputs

def test_valid(model, val):
    data = prep_data(val)
    batches = shuffle_batch(data)
    dev = model.device
    L = 0.0
    for batch in batches:
        x, a, y = batch
        x, a, y = torch.from_numpy(x).to(dev), torch.from_numpy(a).to(dev), torch.from_numpy(y).to(dev)
        # pred = model(x, a, train=True)
        # l = model.loss(pred, y)
        l = model.get_loss(x, a, y)
        L += l.item() * x.shape[0]
    L = L / len(data[0])
    return L

def get_preds(model, val):
    data = prep_data(val)
    batches = shuffle_batch(data)
    dev = model.device
    preds = []
    for batch in batches:
        x, a, y = batch
        x, a, y = torch.from_numpy(x).to(dev), torch.from_numpy(a).to(dev), torch.from_numpy(y).to(dev)
        # pred = model(x, a, train=True)
        # l = model.loss(pred, y)
        pred = model(x, a, train=False)
        preds.extend(pred)
    return preds
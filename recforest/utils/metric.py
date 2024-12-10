import numpy as np

def recall(true: list[list], pred: list[list], normalized: bool = False) -> float:
    recall = 0
    nonempty_cnt = 0
    for p, t in zip(pred, true):
        if len(t) == 0:
            continue
        if normalized:
            recall += len(set(p) & set(t)) / min(len(p), len(t))
        else:
            recall += len(set(p) & set(t)) / len(t)
        nonempty_cnt += 1
    print(f'emtpy user count: {len(true) - nonempty_cnt}')
    recall /= nonempty_cnt
    return recall

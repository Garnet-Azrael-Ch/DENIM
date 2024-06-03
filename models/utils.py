from tensorboardX import SummaryWriter
import torch
# from utils import mapping_ARs_token_index

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


def args_metric(true_args_list, pred_args_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true_args, pred_args in zip(true_args_list, pred_args_list):
        true_args_set = set(true_args)
        pred_args_set = set(pred_args)
        assert len(true_args_set) == len(true_args)
        assert len(pred_args_set) == len(pred_args)
        tp += len(true_args_set & pred_args_set)
        fp += len(pred_args_set - true_args_set)
        fn += len(true_args_set - pred_args_set)
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre * rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre * rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp + fn + 1e-10)
    return {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}

def get_meaningful_ARs_turple(num_span, dataset):

    if dataset=="AAEC":
        list_meaningful_turple = []
        for i in range(num_span - 1):
            for j in range(i+1, num_span):
                list_meaningful_turple.append((i, j))
    elif dataset=="AbstRCT":
        list_meaningful_turple = []
        for i in range(num_span - 1):
            for j in range(i+1, num_span):
                list_meaningful_turple.append((i, j))
    else:
        raise ValueError
    
    return list_meaningful_turple


def mapping_index_to_ARs_token(index, dataset):
    # 0 对应 (0, 1)
    index += 1
    pair_0 = 0
    pair_1 = 1

    if dataset == "AAEC":
        row_num = 11
        while index - row_num > 0:
            index -= row_num
            pair_0 += 1
            pair_1 += 1
            row_num -= 1
        pair_1 += (index - 1)
    elif dataset == "AbstRCT":
        row_num = 10
        while index - row_num > 0:
            index -= row_num
            pair_0 += 1
            pair_1 += 1
            row_num -= 1
        pair_1 += (index - 1)
    else:
        raise ValueError
    
    return (pair_0, pair_1)

class Scorer:
    def __init__(self):
        self.s = 0
        self.g = 0
        self.c = 0
        return

    def add(self, predict, gold):
        self.s += len(predict)
        self.g += len(gold)
        self.c += len(gold & predict)
        return

    @property
    def p(self):
        return self.c / self.s if self.s else 0.

    @property
    def r(self):
        return self.c / self.g if self.g else 0.

    @property
    def f(self):
        p = self.p
        r = self.r
        return (2. * p * r) / (p + r) if p + r > 0 else 0.0

    def dump(self):
        return {
            'g': self.g,
            's': self.s,
            'c': self.c,
            'p': self.p,
            'r': self.r,
            'f': self.f
        }


def eval_edge_PE(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[4] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Support" , 1: "Attack"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in p_sample if edge[4] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in g_sample if edge[4] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

def eval_component_PE(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    # assert len(edge_labels) == 3, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Premise" , 1: "Claim", 2: "MajorClaim"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores


def eval_edge_AbstRCT(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[4] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 3, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Support" , 1: "Partial-Attack", 3: "Attack"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in p_sample if edge[4] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in g_sample if edge[4] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

def eval_component_AbstRCT(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 3, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Evidence" , 1: "Claim", 2: "MajorClaim"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores
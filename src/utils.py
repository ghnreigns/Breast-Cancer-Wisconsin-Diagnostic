import random
from typing import Callable, Dict, List
import numpy as np
from functools import wraps
from time import time
from sklearn import metrics


def set_seeds(seed: int = 1234) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def timing(f: Callable):
    """A decorator to time a function call run.

    Args:
        f ([type]): [description]

    Returns:
        [type]: [description]
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"time taken: {te-ts} sec")
        # print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def add_summary_col(inner_dict: Dict[str, Dict[str, List]]):
    """A utility function to transform model_dict into a dataframe

    Args:
        inner_dict (Dict[str, Dict[str, List]]): [description]
    """
    for k, v in inner_dict.items():
        if k == "identifier":
            inner_dict[k].extend(["mean_cv", "oof_cv"])
        elif k in ["y_true", "y_pred", "y_prob"]:
            inner_dict[k].append(np.concatenate(v))
            inner_dict[k].append(inner_dict[k][-1])
        elif k in ["roc", "brier_loss"]:
            inner_dict[k].extend([sum(v) / len(v), None])
        elif k in ["precision", "recall", "f1"]:
            i0s = [elm[0] for elm in v]
            i1s = [elm[1] for elm in v]
            inner_dict[k].extend([[sum(i0s) / len(i0s), sum(i1s) / len(i1s)], None])
        else:
            inner_dict[k].extend([None, None])
    inner_dict["confusion_matrix"][-2] = metrics.confusion_matrix(
        inner_dict["y_true"][-1], inner_dict["y_pred"][-1]
    )
    inner_dict["confusion_matrix"][-1] = inner_dict["confusion_matrix"][-2].copy()
    inner_dict["roc"][-1] = metrics.roc_auc_score(
        y_true=inner_dict["y_true"][-1], y_score=inner_dict["y_prob"][-1]
    )

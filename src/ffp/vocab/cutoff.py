"""
Frequency Cutoffs
"""
import collections
import operator
from os import PathLike
from typing import Counter, Union


class Cutoff:  # pylint: disable=too-few-public-methods
    """
    Frequency Cutoff

    Defines how a vocabulary is sized, if mode is 'min_freq', items with frequency lower than
    `cutoff` are discarded. If mode is 'target_size', the number of items will be smaller than or
    equal to `cutoff`, discarding items at the next frequency-boundary.
    """
    def __init__(self, cutoff: int, mode: str = "min_freq"):
        self.cutoff = cutoff
        self.mode = mode

    @property
    def mode(self) -> str:
        """
        Return the cutoff mode, one of "min_freq" or "target_size".
        :return: The cutoff mode
        """
        return "min_freq" if self._min_freq else "target_size"

    @mode.setter
    def mode(self, mode: str):
        if mode.lower() == "min_freq":
            self._min_freq = True
        elif mode.lower() == "target_size":
            self._min_freq = False
        else:
            raise ValueError(
                "Unknown cutoff mode, expected 'min_freq' or 'target_size' but got: "
                + mode)


def _count_words(file: Union[str, bytes, int, PathLike]) -> Counter:
    cnt = collections.Counter()
    with open(file) as inf:
        for line in inf:
            for word in line.strip().split():
                cnt[word] += 1
    return cnt


def _filter_and_sort(cnt: Counter, cutoff: Cutoff):
    cutoff_v = cutoff.cutoff

    def cmp(tup):
        return tup[1] >= cutoff_v

    if cutoff.mode == "min_freq":
        items = sorted(filter(cmp, cnt.items()),
                       key=operator.itemgetter(1, 0),
                       reverse=True)
        if not items:
            return [], []
        keys, cnt = zip(*items)
    else:
        keys, cnt = zip(
            *sorted(cnt.items(), key=operator.itemgetter(1, 0), reverse=True))
        if cutoff_v == 0:
            return [], []
        # cutoff is size, but used as idx
        cutoff_v -= 1
        if cutoff_v <= len(cnt) - 2:
            cnt_at_target = cnt[cutoff_v]
            cnt_after_target = cnt[cutoff_v + 1]
            if cnt_at_target == cnt_after_target:
                while cutoff_v > 0 and cnt[cutoff_v] == cnt_after_target:
                    cutoff_v -= 1
        keys = keys[:cutoff_v + 1]
        cnt = cnt[:cutoff_v + 1]
    return list(keys), list(cnt)


__all__ = ['Cutoff']

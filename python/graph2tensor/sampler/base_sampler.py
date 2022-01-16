#!/usr/bin/env python3
from typing import Union, Sequence
import numpy as np


class BaseSampler(object):

    def sample(self, ids: Union[np.ndarray, Sequence[int]], **kwargs):
        raise NotImplemented


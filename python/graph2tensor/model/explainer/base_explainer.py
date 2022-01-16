#!/usr/bin/env python3


class BaseExplainer:

    def explain(self, inputs, target_class):
        raise NotImplemented

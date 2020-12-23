from espnet2.layers.abs_normalize import AbsNormalize
import torch

class Normalize(AbsNormalize):
    def __init__(self):
        super().__init()
        print(0)
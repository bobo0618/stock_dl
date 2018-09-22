from easydict import *
import numpy as np
import simplejson as jason

CFG = EasyDict()
CFG.TRAIN = EasyDict()
CFG.TRAIN.BATCH_SIZE = 32
CFG.TRAIN.EPOCHs = 50
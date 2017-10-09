from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
from onmt.metrics.gleu import sentence_gleu
from onmt.metrics.sbleu import sentence_bleu
from onmt.metrics.bleu import moses_multi_bleu
from onmt.utils import split_batch
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np



import torch.nn as nn
import models
from models import *

def get_model(config):    
    model = getattr(getattr(models, config.model), config.model)(config)
    return model
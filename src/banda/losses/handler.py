
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss

from banda.data.item import SourceSeparationBatch

class LossHandler(_Loss):
    def __init__(self, *, 
                 config: DictConfig):
        super().__init__()
        
        self.config = config
        
        
    def forward(self, batch: SourceSeparationBatch):
        pass
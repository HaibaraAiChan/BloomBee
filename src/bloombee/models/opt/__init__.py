from bloombee.models.opt.block import WrappedOPTBlock  
from bloombee.models.opt.config import DistributedOPTConfig  
from bloombee.models.opt.model import (  
    DistributedOPTForCausalLM,  
    DistributedOPTForSequenceClassification,  
    DistributedOPTModel,  
)  
# from bloombee.models.opt.speculative_model import DistributedOPTForSpeculativeGeneration  
from bloombee.utils.auto_config import register_model_classes  

register_model_classes(  
    config=DistributedOPTConfig,  
    model=DistributedOPTModel,  
    model_for_causal_lm=DistributedOPTForCausalLM,  
    # model_for_speculative=DistributedOPTForSpeculativeGeneration,  
    model_for_sequence_classification=DistributedOPTForSequenceClassification,  
)
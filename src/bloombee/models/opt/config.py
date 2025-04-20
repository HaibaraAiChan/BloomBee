import os  
from typing import Optional, Union  

from hivemind import get_logger  
from transformers.models.opt import OPTConfig  
from transformers.models.opt.modeling_opt import OPTAttention  

from bloombee.client.config import ClientConfig  
from bloombee.client.lm_head import LMHeadConfig  
from bloombee.client.ptune import PTuneConfig  
from bloombee.models.opt.block import WrappedOPTBlock  

logger = get_logger(__name__)  


class DistributedOPTConfig(OPTConfig, ClientConfig, PTuneConfig, LMHeadConfig):  
    model_type = "opt"
    block_class = WrappedOPTBlock  
    attn_class = OPTAttention  
    block_prefix = "decoder.layers"  
    
    num_key_value_groups = 1
    # @property
    # def num_key_value_groups(self) -> int:
    #     if self.new_decoder_architecture:
    #         return self.num_attention_heads // self.num_kv_heads
    #     # if self.multi_query:
    #     #     return self.num_attention_heads
    #     return 1
    
    @classmethod  
    def from_pretrained(  
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs  
    ):  
        logger.info(  
            "Make sure you follow the OPT terms of use: "  
            "https://github.com/facebookresearch/fairseq/blob/main/LICENSE"  
        )  

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)  
        if loading_from_repo and dht_prefix is None:  
            dht_prefix = str(model_name_or_path) 
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts  
            dht_prefix = dht_prefix.replace(".", "-")  
            if not dht_prefix.endswith("-hf"):  
                dht_prefix += "-hf"  
            logger.info(f"Using DHT prefix: {dht_prefix}")  
            
        print('opt/config.py from_pretrained dht_prefix ', dht_prefix)
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)  
        config = result[0] if isinstance(result, tuple) else result  
        # config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by bloombee  
        config.use_cache = False  # use_cache=False leads to identical results but is slower and not supported by bloombee  
        
        return result
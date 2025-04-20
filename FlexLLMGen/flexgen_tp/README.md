To utilize all GPUs on a single compute node for inference with tensor parallelism, use the following:

perform ```pip install e .``` from BloomBee/FlexLLMGen/  directory and run using the following from the same directory:

```
 python3 -m flexgen_tp.flex_opt --model facebook/opt-1.3b --gen-len 32
```
the installation should install flexgen_tp package along with flexllmgen. Delete the downloaded dataset first, if you have run flexllmgen.flex_opt.py

Use the same directory for running flexllmgen also.

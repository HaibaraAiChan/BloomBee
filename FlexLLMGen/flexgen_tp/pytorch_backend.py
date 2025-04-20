"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(100)
import pdb

############# Edited


from flexgen_tp.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

############ Edited
import torch.distributed as dist
import sys


general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen_tp import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):

        ############ Edited
        #torch.cuda.set_device(device)

        ############
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()
        
        ########### EDited
        #print(f'  SELF, device:{device} , self.data:{self.data.shape}, self.device:{self.device}, type:{type(data)}\n\n')
        ##########

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):

        #print('    load_from_np:\n self.device.device_type:{self.device.device_type} ')
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
                #print('\n\nHEREH HERE HERE HERE HERE.\n')

        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
                #print('\n\n\n ELSE HERE HERE HERE HERE.\n')

            else:

                ############Edited
                #print('Y YYYY')
                #breakpoint()

                #t=None

                #t=torch.from_numpy(np_array)
                
                #pdb.set_trace()
                
                #print(f'      t: {t}')
                #print(f'      self.data.shape: {self.data.shape} ,{self.data.dtype}, {self.data.device} ')
                #print(f'       reached copy_:np_array: {np_array.shape} type:{type(np_array)}  device-type:{self.device.device_type}')
                #print(f'   self.data.device:{self.data.device},self.data.shape:{self.data.shape} self.data:{self.data}, type:{type(self.data)}')

                #dist.barrier()
                ############
                #import subprocess
                #nvidia_smi_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
                #print(f'nvidia-smi output before copy:\n{nvidia_smi_output.stdout}')

                #breakpoint()
                #print(f'torch from type:{torch.from_numpy(np_array).device} self.data.type:{type(self.data)}')
                #self.data=torch.from_numpy(np_array)
                #dist.barrier()
                
                #torch.cuda.synchronize()
                #print('sync')
                #self.data.copy_(torch.from_numpy(np_array))
                    
                ############ EDited
                try:
                # Attempt to copy the NumPy array into the PyTorch tensor
                    self.data.copy_(torch.from_numpy(np_array))
                    #print(f'inside try:self.data:{self.data}, \n try self.data.shape:{self.data.shape}')
                    #print(" after self.data.copy_ HEREH HEREH HERE")
                except Exception as e:
                # Print detailed error information
                    print("Exception occurred during tensor copy")
                    print(f"Exception type: {type(e)}")
                    print(f"Exception message: {e}")

                # Print the traceback to understand where the error occurred
                    traceback.print_exc()
                
                #print('          finished data.copy_')
                #dist.barrier()
                ############

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            ######## Edited
            #print('called load_from_np')
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache
    '''
    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        #print(f' wq:{w_q.shape}, hidden:{hidden.shape}, head dim :{head_dim} inputs:{inputs.shape}')
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        ######################### EDITED
        #print(f'inside mha: {attention_mask.data.shape}')
        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        #print(f'causal mask.shape:{causal_mask.shape}')
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        #breakpoint()

        return TorchTensor.create_from_torch(value, self), k, v
        '''

    def mha_TP(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, rank, world_size):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        n_head_sliced = n_head//world_size

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        #print(f' wq:{w_q.shape}, hidden:{hidden.shape}, head dim :{head_dim} inputs:{inputs.shape}')
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head_sliced, head_dim)
        k = k.view(b, s, n_head_sliced, head_dim)
        v = v.view(b, s, n_head_sliced, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head_sliced, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)
        
        #print(f'v after permute:{v.shape}')
        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        ######################### EDITED
        #print(f'inside mha: {attention_mask.data.shape}')
        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        #print(f'causal mask.shape:{causal_mask.shape}')
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head_sliced, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head_sliced, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)

        #print(f'attn_weights shape:{attn_weights.shape}')
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head_sliced, s, head_dim)
        
        #print(f'value shape:{value.shape}')

        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, -1)

        ######### issue with previous(original line) is that during run, it is not contiguous, so reshape, or use contiguous
        #value = value.transpose(1, 2).contiguous().reshape(b, s, -1)
        value = F.linear(value, w_out.data, bias=b_out.data)

        ######## all gathering for value

        dist.all_reduce(value, op = dist.ReduceOp.SUM)

        value.add_(inputs.data)
        ####################

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        
        # now n_head_sliced, so ,(s, b * n_head_sliced, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        #print(f'prefill shape:k: {k.shape}, v:{v.shape} s:{s}, b:{b}, n_head_sliced:{n_head_sliced} head_dim:{head_dim}')

        '''
        **** exploring slicing kv cache for different GPU, in tensor parallelism
        '''

                
        ############ check here if this is working
        ######## this is to align, as it was sliced,
        ######## for further, check manual tensor parallel work comments

        k = k.contiguous().view(s,b,-1)
        v = v.contiguous().view(s,b,-1)
        
        ########### all gather for k & v

        all_k_data = [torch.zeros_like(k) for _ in range(world_size)]
        dist.all_gather(all_k_data, k)
        # s, b, n_head* head_dim
        k = torch.cat(all_k_data, dim = 2)
        k = k.view(s, b * n_head, head_dim)
        
        all_v_data = [torch.zeros_like(v) for _ in range(world_size)]
        dist.all_gather(all_v_data, v)
        v = torch.cat(all_v_data, dim = 2)
        v = v.view(s, b * n_head, head_dim)
        ##https://stackoverflow.com/questions/74725908/distributed-torch-data-collision-from-all-gather-writing-all-gather-results-to
        
        dist.barrier()
        ##############
        
        
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        #breakpoint()
        #print(f'\n inside mha_TP k:{k.shape}, v:{v.shape}')
        return TorchTensor.create_from_torch(value, self), k, v



    '''
    def mha_tp_manual(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        

        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        
        #pass

        cuda_device0 = torch.device("cuda:0") 
        cuda_device1 = torch.device("cuda:1")
        WORLD_SIZE = 2

        ## transposing weights because F.linear transposes weights before doing dot product with input matrix
        w_q = w_q.data.T
        w_k = w_k.data.T
        w_v = w_v.data.T
        w_out = w_out.data.T
        b_q = b_q.data
        b_k = b_k.data
        b_v = b_v.data
        b_out = b_out.data

        w_ln = w_ln.data
        b_ln = b_ln.data

        # rank 0
        attention_mask0 = attention_mask.data.to(cuda_device0)
        # rank 1
        attention_mask1 = attention_mask.data.to(cuda_device1)


        col_splits_num = int(w_q.shape[1]/WORLD_SIZE) 
        row_splits_num = int(w_out.shape[0]/WORLD_SIZE) 
    
        zero = 0
        one = 1
        # suppletion
        # rank 0

        wq0 = w_q[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        wk0 = w_k[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        wv0 = w_v[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        
        wo0 = w_out[(zero * row_splits_num): ((zero * row_splits_num) + row_splits_num), :].to(cuda_device0)
        
        # rank 1
        wq1 = w_q[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        wk1 = w_k[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        wv1 = w_v[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)

        wo1 = w_out[(one * row_splits_num): ((one * row_splits_num) + row_splits_num), :].to(cuda_device1)

        # rank 0
        b_col_q0 = b_q[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)    
        b_col_k0 = b_k[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        b_col_v0 = b_v[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        bo0 = (b_out/WORLD_SIZE).to(cuda_device0)

        # rank 1
        b_col_q1 = b_q[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device1)            
        b_col_k1 = b_k[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device1)
        b_col_v1 = b_v[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device1)
        bo1 = (b_out/WORLD_SIZE).to(cuda_device1)
        
        # rank 0
        x0 = inputs.data.to(cuda_device0)
        w_ln0 = w_ln.to(cuda_device0)
        b_ln0 = b_ln.to(cuda_device0)
        
        # rank 1
        x1 = inputs.data.to(cuda_device1)
        w_ln1 = w_ln.to(cuda_device1)
        b_ln1 = b_ln.to(cuda_device1)
        
        b,s,h = inputs.shape

        head_dim = int(h/n_head)
        
        start_time = time.perf_counter()
        # rank 0
        hidden0 = F.layer_norm(x0, (x0.shape[2],), weight = w_ln0.data, bias = b_ln0.data)        
        # rank 1
        hidden1 = F.layer_norm(x1, (x1.shape[2],), weight = w_ln1.data, bias = b_ln1.data)


        hidden_shape = hidden0.shape
        # rank 0
        hidden0 = hidden0.view(-1, hidden0.shape[2])
        # rank 1
        hidden1 = hidden1.view(-1, hidden1.shape[2])


        scaling = head_dim ** -0.5
        
        # rank 0
        q0 = (torch.mm(hidden0, wq0) + b_col_q0 ) * scaling
        k0 = torch.mm(hidden0, wk0) + b_col_k0
        v0 = torch.mm(hidden0, wv0) + b_col_v0 
        
        # rank 1
        q1 = (torch.mm(hidden1, wq1) + b_col_q1 ) * scaling
        k1 = torch.mm(hidden1, wk1) + b_col_k1
        v1 = torch.mm(hidden1, wv1) + b_col_v1

        # putting in 3d again before 4d conversion
        # rank 0
        q0 = q0.view(b, s, -1)
        k0 = k0.view(b, s, -1)
        v0 = v0.view(b, s, -1)
        
        # rank 1
        q1 = q1.view(b, s, -1)
        k1 = k1.view(b, s, -1)
        v1 = v1.view(b, s, -1)

        #print(f'rank:{LOCAL_RANK}, q:\n{q}, k:\n{k}, v:\n{v}')

        #********* very important, we slice heads, rather than head_dim
        n_head_sliced = int(n_head/WORLD_SIZE)
        
        # rank 0
        q0 = q0.view(b, s, n_head_sliced, head_dim)
        k0 = k0.view(b, s, n_head_sliced, head_dim)
        v0 = v0.view(b, s, n_head_sliced, head_dim)
        
        # rank 1
        q1 = q1.view(b, s, n_head_sliced, head_dim)
        k1 = k1.view(b, s, n_head_sliced, head_dim)
        v1 = v1.view(b, s, n_head_sliced, head_dim)

        #print(f'rank:{LOCAL_RANK}, q.shape:{q.shape}')
        
        # rank 0
        q0 = q0.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)
        # shape: (b * n_head_sliced, head_dim, s)
        k0 = k0.permute(0, 2, 3, 1).reshape(b * n_head_sliced, head_dim, s)
        # shape: (b * n_head_sliced, s, head_dim)
        v0 = v0.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)
        
        # rank 1
        q1 = q1.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)
        # shape: (b * n_head_sliced, head_dim, s)
        k1 = k1.permute(0, 2, 3, 1).reshape(b * n_head_sliced, head_dim, s)
        # shape: (b * n_head_sliced, s, head_dim)
        v1 = v1.permute(0, 2, 1, 3).reshape(b * n_head_sliced, s, head_dim)

        # rank 0
        attn_weights0 = torch.bmm(q0, k0)
        # rank 1
        attn_weights1 = torch.bmm(q1, k1)
        
        #print(f'rank:{LOCAL_RANK} attn_weights before idx:{attn_weights}')
        

        ## rank 0
        # shape: (b, 1, s, s)
        idx0 = torch.arange(s)
        causal_mask0 = (idx0 <= idx0.view(s, 1)).view(1, 1, s, s).to(cuda_device0)
        mask0 = attention_mask0.data.view(b, 1, 1, s).bool() & causal_mask0

        # shape: (b, n_head, s, s)
        attn_weights0 = attn_weights0.view(b, n_head_sliced, s, s)
        attn_weights0 = torch.where(mask0, attn_weights0, -1e4)
        attn_weights0 = attn_weights0.view(b * n_head_sliced, s, s)
        attn_weights0 = F.softmax(attn_weights0, dim=2)
        # shape: (b, n_head, s, head_dim)
        value0 = torch.bmm(attn_weights0, v0).view(b, n_head_sliced, s, head_dim)
        
        ## rank 1
        # shape: (b, 1, s, s)
        idx1 = torch.arange(s)
        causal_mask1 = (idx1 <= idx1.view(s, 1)).view(1, 1, s, s).to(cuda_device1)
        mask1 = attention_mask1.data.view(b, 1, 1, s).bool() & causal_mask1

        # shape: (b, n_head, s, s)
        attn_weights1 = attn_weights1.view(b, n_head_sliced, s, s)
        attn_weights1 = torch.where(mask1, attn_weights1, -1e4)
        attn_weights1 = attn_weights1.view(b * n_head_sliced, s, s)
        attn_weights1 = F.softmax(attn_weights1, dim=2)
        # shape: (b, n_head, s, head_dim)
        value1 = torch.bmm(attn_weights1, v1).view(b, n_head_sliced, s, head_dim)

        #print(f'rank:{LOCAL_RANK}, value:{value.shape}')

        # shape: (b, s, h)
        #value = value.transpose(1, 2).reshape(b, s, h)
        

        # orignially value.transpose(1,2).reshape(b, s, num_head * head_dim)
        # but we are having num_head_sliced for each GPU
        # so , we would do (b,s, -1), -1 for num_head_sliced * head_dim
        
        # rank 0
        value0 = value0.transpose(1, 2).reshape(b, s, -1)
        # rank 1
        value1 = value1.transpose(1, 2).reshape(b, s, -1)

        # again making it 2d
        value_shape = value0.shape

        # rank 0
        value0 = value0.view(-1, value0.shape[2])
        value0 = torch.mm(value0, wo0) + bo0
        # rank 1
        value1 = value1.view(-1, value1.shape[2])
        value1 = torch.mm(value1, wo1) + bo1


        # putting that back
        # -1 for wo.shape[1]
        # rank 0
        value0 = value0.view(value_shape[0], value_shape[1], -1)
        # rank 1 
        value1 = value0.view(value_shape[0], value_shape[1], -1)

        

        # ********* adding input should be done after All-reduce otherwise, we would add 2* inputs 
        #value.add_(x)

        # originally: (s, b * n_head, head_dim)
        # but we got : (s, b * n_head_sliced, head_dim)
        
        # rank 0
        k0 = k0.permute(2, 0, 1)
        v0 = v0.permute(1, 0, 2)
        # rank 1

        # to cuda_device0 to manually all-gather them
        k1 = k1.permute(2, 0, 1).to(cuda_device0)
        v1 = v1.permute(1, 0, 2).to(cuda_device0)
        
        #******** this step is extra, so that in the last dimension we can gather the head values
        #******** otherwise heads are alternated(try removing the following 2 line and print to check)
       

        # print(f's:{s}, b:{b},n_head_sliced:{n_head_sliced}, n_head:{n_head}, k0:{k0.shape}, v0:{v0.shape}')        
        # s,b, n_head_sliced * head_dim
        k0 = k0.contiguous().view(s,b,-1)
        v0 = v0.contiguous().view(s,b,-1)
        k1 = k1.contiguous().view(s,b,-1)
        v1 = v1.contiguous().view(s,b,-1)

        k_shape = k0.shape
        v_shape = v0.shape

        #k=k.contiguous().view(-1, k.shape[2])
        #v=v.contiguous().view(-1, v.shape[2])

        #print(f'rank:{LOCAL_RANK}, value shape:{value.shape}, k:{k.shape}, v:{v.shape}')
        
        # all reduce, & all gather done manually
        current_device_index = torch.cuda.current_device()
        current_device = torch.device(f"cuda:{current_device_index}")
        # all reduce
        value = value0 + value1.to(cuda_device0)
        value.add_(inputs.data)
        # all gather to fix shape

        # s, b, n_head*head_dim
        k = torch.cat((k0,k1), dim = 2)
        v = torch.cat((v0,v1), dim = 2)
        # s, b*n_head, head_dim
        k = k.view(s, b*n_head, head_dim)
        v = v.view(s, b*n_head, head_dim)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        #torch.cuda.synchronize()
        end_time = time.perf_counter()
        compute_time = end_time - start_time

        return TorchTensor.create_from_torch(value, self), TorchTensor.create_from_torch(k, self), TorchTensor.create_from_torch(v, self), compute_time
        '''
        
    '''
    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights

        #print('MHA_GEN CALLED.\n')
        #############
        #print(f'\n compress_cache:{compress_cache}\n')


        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        #print(f' b:{b}, tgt_s:{tgt_s}, h:{h}, src_s:{src_s}, head_dim:{head_dim} n_head:{n_head}')

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)

        #print(f'before linear: q:{q.shape}, k:{k.shape}, v:{v.shape}')

        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)
        
        #print(f'after linear: q:{q.shape}, k:{k.shape}, v:{v.shape}')

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        
        #print(f'after permute: q:{q.shape}, k_new:{k_new.shape}, v_new:{v_new.shape}')

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                    print(f'cache: k_cache:{k_cache.shape}, v_cache:{v_cache.shape}')
                    print(f'k shape: {k.shape}, v.shape:{v.shape}')

                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        
        #breakpoint()

        #print(f'TYPE:{type(k_new)}, value:{value.shape}, k_new shape:{k_new.shape}, v_new:{v_new.shape}')
        

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    '''

    def mha_gen_TP(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, rank, world_size):
        """Multi-head attention (decoding phase)."""
        # decompress weights

        #print('MHA_GEN CALLED.\n')
        #############
        #print(f'\n compress_cache:{compress_cache}\n')


        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        #print(f' b:{b}, tgt_s:{tgt_s}, h:{h}, src_s:{src_s}, head_dim:{head_dim} n_head:{n_head}')

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)

        #print(f'before linear: q:{q.shape}, k:{k.shape}, v:{v.shape}')
        
        n_head_slice=int(n_head/world_size)
        #print(f'n_head:{n_head}, world_size:{world_size}, b:{b}, n_head_slice:{n_head_slice}')

        q = q.view(b, tgt_s, n_head_slice, head_dim)
        k = k.view(b, tgt_s, n_head_slice, head_dim)
        v = v.view(b, tgt_s, n_head_slice, head_dim)
        
        #print(f'    mha_gen_TP linear: q:{q.shape}, k:{k.shape}, v:{v.shape}')

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head_slice, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head_slice, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head_slice, head_dim)
        
        #print(f'    mha_gen_TP after permute: q:{q.shape}, k_new:{k_new.shape}, v_new:{v_new.shape}')
        

        # this is what must be done, becaue we can not slice full KV cache for tensor parallelism in this code
        ## we do not need the full, need the half, but we are getting the full
        k_cache_data = k_cache.data.view(k_cache.shape[0], b, n_head*head_dim)
        v_cache_data = v_cache.data.view(v_cache.shape[0], b, n_head*head_dim)

        kv_split_num = int(k_cache_data.shape[2]/world_size)

        # splitting kv_cache for findividual GPU
        k_cache_col = k_cache_data[:,:, (rank * kv_split_num): (rank * kv_split_num + kv_split_num)]
        v_cache_col = v_cache_data[:,:, (rank * kv_split_num): (rank * kv_split_num + kv_split_num)]
        
        k_cache_col=k_cache_col.contiguous().view(-1, b*n_head_slice, head_dim)
        v_cache_col=v_cache_col.contiguous().view(-1, b*n_head_slice, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache_col)[:src_s]
                    v = v_cache.device.decompress(v_cache_col)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache_col[:src_s]
                    v = v_cache_col[:src_s]
                    #print(f'cache: k_cache:{k_cache.shape}, v_cache:{v_cache.shape}')
                    #print(f'k shape: {k.shape}, v.shape:{v.shape}')

                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head_slice, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head_slice, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head_slice, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head_slice, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache_col[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head_slice, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head_slice, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head_slice, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)
        
        # original
        # shape: (b, 1, h)
        #value = value.transpose(1, 2).view(b, tgt_s, h)
        
        #because of slicing, h=(head_dim * (n_head_slice * world_size)
        value = value.transpose(1,2).view(b, tgt_s, n_head_slice * head_dim) 

        value = F.linear(value, w_out.data, bias=b_out.data)
        

        ######## all reduce
        dist.all_reduce(value,  op = dist.ReduceOp.SUM)
        ####### 

        value.add_(inputs.data)


        ### WRITE ALL _Gather for k v,
        k_new = k_new.contiguous().view(tgt_s,b,-1)
        v_new = v_new.contiguous().view(tgt_s,b,-1)
        
        ########### all gather for k & v
         
        all_k_data = [torch.zeros_like(k_new) for _ in range(world_size)]
        dist.all_gather(all_k_data, k_new)
        # s, b, n_head* head_dim
        k_new = torch.cat(all_k_data, dim = 2)
        k_new = k_new.view(tgt_s, b * n_head, head_dim) 
        
        all_v_data = [torch.zeros_like(v_new) for _ in range(world_size)]
        dist.all_gather(all_v_data, v_new)
        v_new = torch.cat(all_v_data, dim = 2) 
        v_new = v_new.view(tgt_s, b * n_head, head_dim)
        ##########


        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        
        #breakpoint()

        #print(f'TYPE:{type(k_new)}, value:{value.shape}, k_new shape:{k_new.shape}, v_new:{v_new.shape}')
        

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    '''
    def mha_gen_2(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,attn_sparsity, compress_cache, comp_config):
        
        cuda_device0 = torch.device("cuda:0") #("cuda:{}".format(LOCAL_RANK))
        cuda_device1 = torch.device("cuda:1")
        
        # rank 0
        attention_mask0=attention_mask.data.to(cuda_device0)
        # rank 1
        attention_mask1=attention_mask.data.to(cuda_device1)

        #inputs=inputs.data
        b, tgt_s, h = inputs.shape
        src_s = attention_mask.data.shape[1]
        #print(f'attention_mask:{attention_mask.shape}')

        w_q = w_q.data.T
        w_k = w_k.data.T
        w_v = w_v.data.T
        w_out = w_out.data.T
        b_q = b_q.data
        b_k = b_k.data
        b_v=b_v.data
        b_out=b_out.data
        

        w_ln=w_ln.data
        b_ln=b_ln.data

        #k_cache=k_cache.data
        #v_cache=v_cache.data

        WORLD_SIZE=2

        col_splits_num = int(w_q.shape[1]/WORLD_SIZE) 
        row_splits_num = int(w_out.shape[0]/WORLD_SIZE) 
        

        zero=0
        one=1
        # suppletion
        # rank 0
        wq0 = w_q[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        wk0 = w_k[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        wv0 = w_v[:,(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        
        wo0 = w_out[(zero * row_splits_num): ((zero * row_splits_num) + row_splits_num), :].to(cuda_device0)
        
        #rank 1
        wq1 = w_q[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        wk1 = w_k[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        wv1 = w_v[:,(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)

        wo1 = w_out[(one * row_splits_num): ((one * row_splits_num) + row_splits_num), :].to(cuda_device1)

        #rank 0
        b_col_q0 = b_q[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)    
        b_col_k0 = b_k[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        b_col_v0 = b_v[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)

        #rank 1
        b_col_q1 = b_q[(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        b_col_k1 = b_k[(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        b_col_v1 = b_v[(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
    
        #rank 0
        x0 = inputs.data.to(cuda_device0)
        w_ln0 = w_ln.to(cuda_device0)
        b_ln0 = b_ln.to(cuda_device0)
        bo0 = (b_out/WORLD_SIZE).to(cuda_device0)
        
        #rank 1
        x1 = inputs.data.to(cuda_device1)
        w_ln1 = w_ln.to(cuda_device1)
        b_ln1 = b_ln.to(cuda_device1)
        bo1 = (b_out/WORLD_SIZE).to(cuda_device1)


        #scaling =  head_dim ** -0.5

        head_dim = int(h/n_head)
        scaling =  head_dim ** -0.5

        # shape: cache_row, b, h
        # still working in CPU for kv cache
        k_cache = k_cache.data.view(k_cache.shape[0], b, n_head*head_dim)
        v_cache = v_cache.data.view(v_cache.shape[0], b, n_head*head_dim)

        kv_split_num = int(k_cache.shape[2]/WORLD_SIZE)
        
        # rank 0
        k_cache_col0 = k_cache[:,:,(zero * kv_split_num): (zero * kv_split_num + kv_split_num)].to(cuda_device0)
        v_cache_col0 = v_cache[:,:,(zero * kv_split_num): (zero * kv_split_num + kv_split_num)].to(cuda_device0)
        
        # splitting kv_cache for findividual GPU
        # rank 1 
        k_cache_col1 = k_cache[:,:, (one * kv_split_num): (one * kv_split_num + kv_split_num)].to(cuda_device1)
        v_cache_col1 = v_cache[:,:, (one * kv_split_num): (one * kv_split_num + kv_split_num)].to(cuda_device1)
        
        
        #print(f' k_cache_col0 size:{k_cache_col0.shape} v_cache_col0 size:{v_cache_col0.shape}')
        
        #k_cache = k_cache.to(cuda_device)
        #v_cache = v_cache.to(cuda_device)
        
        # measuring computation time here.

        start_time = time.perf_counter()

        #rank0
        hidden0 = F.layer_norm(x0, (x0.shape[2],), weight = w_ln0.data, bias = b_ln0.data)        
        #rank1
        hidden1 = F.layer_norm(x1, (x1.shape[2],), weight = w_ln1.data, bias = b_ln1.data)
        
        hidden_shape = hidden0.shape
        
        #rank 0
        hidden0 = hidden0.view(-1, hidden0.shape[2])
        #rank 1
        hidden1 = hidden1.view(-1, hidden1.shape[2])

        #print(f'h:{h} n_head:{n_head}')

        # shape: (b, 1, h)
        #scaling =  head_dim ** -0.5

        # rank 0
        q0 = (torch.mm(hidden0, wq0) + b_col_q0 ) * scaling
        # rank 1
        q1 = (torch.mm(hidden1, wq1) + b_col_q1 ) * scaling

        #print(f'GPU q:{q}')
        # rank 0
        k0 = torch.mm(hidden0, wk0) + b_col_k0 
        v0 = torch.mm(hidden0, wv0) + b_col_v0 
        #print(f'GPU q :{q} w_q:{wq} b_col_q:{b_col_q}')
        
        # rank 1
        k1 = torch.mm(hidden1, wk1) + b_col_k1
        v1 = torch.mm(hidden1, wv1) + b_col_v1

        #print(f'before view q.shape:{q.shape}')

        #rank 0

        q0 = q0.view(hidden_shape[0], hidden_shape[1], wq0.shape[1])
        k0 = k0.view(hidden_shape[0], hidden_shape[1], wk0.shape[1])
        v0 = v0.view(hidden_shape[0], hidden_shape[1], wv0.shape[1])
        
        # rank 1
        q1 = q1.view(hidden_shape[0], hidden_shape[1], wq1.shape[1])
        k1 = k1.view(hidden_shape[0], hidden_shape[1], wk1.shape[1])
        v1 = v1.view(hidden_shape[0], hidden_shape[1], wv1.shape[1])
        
        # shape: (b, 1, n_head, head_dim)
        
        #head_dim = int(head_dim/WORLD_SIZE)
        #print('head_dim :', head_dim)
        n_head=int(n_head/WORLD_SIZE)

        # rank 0
        q0 = q0.view(b, tgt_s, n_head, -1)
        k0 = k0.view(b, tgt_s, n_head, head_dim)
        v0 = v0.view(b, tgt_s, n_head, head_dim)
    
        q0 = q0.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        k_new0 = k0.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_new0 = v0.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        
        # rank 1
        q1 = q1.view(b, tgt_s, n_head, -1)
        k1 = k1.view(b, tgt_s, n_head, head_dim)
        v1 = v1.view(b, tgt_s, n_head, head_dim)

        q1 = q1.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        k_new1 = k1.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        v_new1 = v1.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)


        #print(f'GPU q , rank{LOCAL_RANK} : \n{q}, rank:{LOCAL_RANK} q.shape:{q.shape}')
        #print(f'GPU k_new, rank:{LOCAL_RANK}: \n{k_new}, rank:{LOCAL_RANK} , k_new shape:{k_new.shape}')
        #print(f'GPU v_new, rank:{LOCAL_RANK}: \n{v_new}, rank :{LOCAL_RANK}, v_shape:{v_new.shape}')

        if attn_sparsity >= 1.0:  # Dense attention
            
            # rank 0
            #making it like previous
            k_cache_col0 = k_cache_col0.contiguous().view(-1, b*n_head, head_dim)
            v_cache_col0 = v_cache_col0.contiguous().view(-1, b*n_head, head_dim)

            k0 = k_cache_col0[:src_s]
            v0 = v_cache_col0[:src_s]
            
            k0[src_s - 1:src_s] = k_new0
            v0[src_s - 1:src_s] = v_new0

            # shape: (b * n_head, head_dim, s)
            k0 = k0.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
            v0 = v0.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)
            
            ##########
            #rank 1
            k_cache_col1 = k_cache_col1.contiguous().view(-1, b*n_head, head_dim)
            v_cache_col1 = v_cache_col1.contiguous().view(-1, b*n_head, head_dim)

            k1 = k_cache_col1[:src_s]
            v1 = v_cache_col1[:src_s]
            
            k1[src_s - 1:src_s] = k_new1
            v1[src_s - 1:src_s] = v_new1

            # shape: (b * n_head, head_dim, s)
            k1 = k1.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
            v1 = v1.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

        else:
            print('sparse attention not implemented\n')
        
        #print(f'value shape:{v.shape}, q.shape:{q.shape}, k shape:{k.shape}, attn.shape:{attention_mask.shape}')
        #print(f'value shape:{v.device}, q.shape:{q.device}, k shape:{k.device}, attention:{attention_mask.device}')
        
        #rank 0
        value0 = self._attention_value(q0, k0, v0, attention_mask0, \
                        b, src_s, tgt_s, n_head, head_dim)
        

        #rank 1
        value1 = self._attention_value(q1, k1, v1, attention_mask1, \
                        b, src_s, tgt_s, n_head, head_dim)

        # rank 0
        value0 = value0.transpose(1, 2).view(b, tgt_s, -1)
        value_shape = value0.shape
        value0 = value0.view(-1, value_shape[2])
        # rank 1
        value1 = value1.transpose(1, 2).view(b, tgt_s, -1)
        value_shape = value1.shape
        value1 = value1.view(-1, value_shape[2])
        
        # rank 0
        output0 = torch.mm(value0, wo0) + bo0
        output0 = output0.view(value_shape[0], value_shape[1], -1)
        
        # rank 1
        output1 = torch.mm(value1, wo1) + bo1
        output1 = output1.view(value_shape[0], value_shape[1], -1)
        
        # rank 0
        output = output0 + output1.to(output0.device)

        # rank 1

        # return TorchTensor.create_from_torch(value, self), k_new, v_new
        current_device_index = torch.cuda.current_device()

        # Create a torch.device object from the current device index
        current_device = torch.device(f"cuda:{current_device_index}")
       

        ##### we are getting k_new0 and k_new1 in different gpu, so putting them together.
        k_new1_back_to_cuda0=k_new1.to(cuda_device0)
        v_new1_back_to_cuda0=v_new1.to(cuda_device0)

        k_new_for_next_layer=torch.concat((k_new0, k_new1_back_to_cuda0), dim=1)
        v_new_for_next_layer=torch.concat((v_new0, v_new1_back_to_cuda0), dim=1)

        # rank 0
        #k_new0 = TorchTensor.create_from_torch(k_new0, self)
        #v_new0 = TorchTensor.create_from_torch(v_new0, self)
        k_new_for_next_layer= TorchTensor.create_from_torch(k_new_for_next_layer, self)
        v_new_for_next_layer= TorchTensor.create_from_torch(v_new_for_next_layer, self)
        
        #output=TorchTensor.create_from_torch(output, self)

        output.add_(inputs.data)
        #torch.cuda.synchronize() 
        end_time = time.perf_counter()
        compute_time = end_time - start_time

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # rank 1

        ########## CAN"T BE DONE :
        #### because Torch.dev is still in GPU


        # k_new1 = TorchTensor.create_from_torch(k_new1, self)
        # v_new1 = TorchTensor.create_from_torch(v_new1, self)

        # print(f'DONE TILL HERE, value:{output.shape}, k_new1 device:{k_new1.device} k_new_for_next_layer shape:{k_new_for_next_layer.shape}, v_new:{v_new_for_next_layer.shape}')
       
        if cuda_device0 == current_device:
            #k_new0 = TorchTensor.create_from_torch(k_new0, self)
            #v_new0 = TorchTensor.create_from_torch(v_new0, self)

            return TorchTensor.create_from_torch(output, self), k_new_for_next_layer, v_new_for_next_layer, compute_time # k_new0, v_new0
        else:
            pass
            #return TorchTensor.create_from_torch(output.to(cuda_device1), self), k_new1, v_new1
    '''

    def _attention_weights(self, q, k, mask, b, src_s, n_head):

        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp_TP(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)
        

        #print(f'in MLP changed return type, return:{type(out)}, max: {torch.max(out)}, min: {torch.min(out)} ')
        dist.all_reduce(out, op = dist.ReduceOp.SUM)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()

        #print(f'in MLP changed return type, return:{type(out)}, max: {torch.max(out)}, min: {torch.min(out)} ')
        #dist.all_reduce(out, op = dist.ReduceOp.SUM)
        
        #print(f'out: {out.shape}, max: {torch.max(out)}, min: {torch.min(out)}, ')
        
        #print(f'out: {out}, max: {torch.max(out[torch.isfinite(out)])}, min: {torch.min(out[torch.isfinite(out)])}, ')
        
        #print(f'contains NaN: {torch.isnan(out).any().item()}, contains Inf: {torch.isinf(out).any().item()}')
        
        #print(f'out: {out}')
        

        #if (torch.isnan(out).any()):
        #    print(f'\n nan found: {torch.isnan(out).any().item()}')
        #    sys.exit(1)
        #if (torch.isinf(out).any()):
        #    print(f'\n inf found: {torch.isinf(out).any().item()}')
        #    sys.exit(1)


        ######## original
        return TorchTensor.create_from_torch(out, self)


        ########## Edited
        #return out, self

    def mlp_2(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):

        ### Still not tested 
        
        #print('called mlp_2')

        cuda_device0 = torch.device("cuda:0") #("cuda:{}".format(LOCAL_RANK))
        cuda_device1 = torch.device("cuda:1")

        b, s, h = inputs.shape
        

        ## we need to transpose(.T), because F.linear takes Wi, Wo and tranposes them before doing dot product with input
        wi=wi.data.T
        wo=wo.data.T
        bi=bi.data
        bo=bo.data
        
        zero = 0
        one = 1
        WORLD_SIZE = 2
        
        col_splits_num = int(wi.shape[1] / WORLD_SIZE)
        row_splits_num = int(wo.shape[0] / WORLD_SIZE)

        # rank 0
        wi0 = (wi[:,(zero * col_splits_num):( (zero * col_splits_num) + col_splits_num)]).to(cuda_device0)
        wo0 = (wo[(zero * row_splits_num): ((zero * row_splits_num) + row_splits_num), :]).to(cuda_device0)
        
        # rank 1
        wi1 = (wi[:,(one * col_splits_num):( (one * col_splits_num) + col_splits_num)]).to(cuda_device1)
        wo1 = (wo[(one * row_splits_num): ((one * row_splits_num) + row_splits_num), :]).to(cuda_device1)
        
        # rank 0
        bi_col0 = bi[(zero * col_splits_num): (zero * col_splits_num + col_splits_num)].to(cuda_device0)
        bo_row0 = (bo/WORLD_SIZE).to(cuda_device0)

        # rank 1
        bi_col1 = bi[(one * col_splits_num): (one * col_splits_num + col_splits_num)].to(cuda_device1)
        bo_row1 = (bo/WORLD_SIZE).to(cuda_device1)
        
        # rank 0
        out0 = F.layer_norm(inputs.data, (h,), weight = w_ln.data, bias = b_ln.data).to(cuda_device0)
        # rank 1
        out1 = F.layer_norm(inputs.data, (h,), weight = w_ln.data, bias = b_ln.data).to(cuda_device1)
        

        # only measuring computation time and returning it

        start_time = time.perf_counter()

        # rank 0
        out0 = out0.view(-1, out0.shape[2])
        out0 = torch.mm(out0, wi0)+ bi_col0
        
        F.relu(out0, inplace=True)

        z0 = torch.mm(out0, wo0) + bo_row0
        z0 = z0.view(-1, inputs.data.shape[1], wo.shape[1])
        
        # rank 1
        out1 = out1.view(-1, out1.shape[2])
        out1 = torch.mm(out1, wi1)+ bi_col1
        
        F.relu(out1, inplace=True)

        z1 = torch.mm(out1, wo1) + bo_row1

        z1 = z1.view(-1, inputs.data.shape[1], wo.shape[1])
        
        #print(f'inside mlp_2: z1.device:{z1.device}')

        # write code for the all reduce
        output = z0 + z1.to(cuda_device0)
        output.add_(inputs.data)
        
        end_time = time.perf_counter()
        #torch.cuda.synchronize()

        computation_time = end_time - start_time

        if donate[0]: inputs.delete()

        return TorchTensor.create_from_torch(output, self), computation_time

        
    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
            print(f'\n\nself.dev:{self.dev} cur_mem:{cur_mem}, peak_mem:{peak_mem}\n\n')

        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()

"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm
import json

@dataclasses.dataclass(frozen=True)
class OptConfig:
    name: str = "opt-125m"
    num_hidden_layers: int = 12
    max_seq_len: int = 2048
    hidden_size: int = 768
    n_head: int = 12
    input_dim: int = 768
    ffn_embed_dim: int = 3072
    pad: int = 1
    activation_fn: str = 'relu'
    vocab_size: int = 50272
    layer_norm_eps: float = 0.00001
    pad_token_id: int = 1
    dtype: type = np.float16

    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_opt_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    # Handle opt-iml-30b and opt-iml-max-30b
    if "-iml-max" in name:
        arch_name = name.replace("-iml-max", "")
    elif "-iml" in name:
        arch_name = name.replace("-iml", "")
    else:
        arch_name = name

    if arch_name == "opt-125m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=12, n_head=12,
            hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
        )
    elif arch_name == "opt-350m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=16,
            hidden_size=1024, input_dim=1024, ffn_embed_dim=1024 * 4,
        )
        raise NotImplementedError("Not implemented because this model "
                                  "has a different architecture")
    elif arch_name == "opt-1.3b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=2048 * 4,
        )
    elif arch_name == "opt-2.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, input_dim=2560, ffn_embed_dim=2560 * 4,
        )
    elif arch_name == "opt-6.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=4096 * 4,
        )
    elif arch_name == "opt-13b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=40, n_head=40,
            hidden_size=5120, input_dim=5120, ffn_embed_dim=5120 * 4,
        )
    elif arch_name == "opt-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4,
        )
    elif arch_name == "galactica-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4, vocab_size=50000,
        )
    elif arch_name == "opt-66b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=64, n_head=72,
            hidden_size=9216, input_dim=9216, ffn_embed_dim=9216 * 4,
        )
    elif arch_name == "opt-175b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=96, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    elif arch_name == "opt-175b-stage":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_opt_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "galactica" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


############# Edited
def download_opt_weights_ORIGINAL(model_name, path, world_size, rank_path=""):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))


############# Edited
def list_files_in_folder(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        file_name_with_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_name_with_path):
            files.append(file_name_with_path)
    return files

def save_tensor_slice(slice_tensor, folder_path, file_name):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the slice as a .npy file
    slice_file_path = os.path.join(folder_path, file_name)
    np.save(slice_file_path, slice_tensor)

def slice_tensors(file_list, json_data, output_base_path, world_size, attn=False):
    
    # file_name : contains full path
    # ouput_base_path: writing location output_base_path/rank i/
    print(f'inside function len:{len(file_list)}')
    print(type(file_list))
    #print(file_list)

    for file_name in file_list.copy():
    #for file_index in range(len(file_list)):
        
        #file_name=file_list[file_index]
        #print("file_name: ", file_name)

        for key in json_data.keys():
            key=key.strip()
            file_name=file_name.strip()

            #print(f'inside function key:{key} f:{file_name} {file_name.endswith(key)}') 
            # all attn weights & biases has self_attn+ ... as file name
            if attn == True:
                file_ending_in_key = 'self_attn' + key
            else:
                file_ending_in_key = key

            if file_name.endswith(file_ending_in_key):
            #if file_name.endswith(key):

                print(f"File '{file_name}' matches key '{key}' with value '{json_data[key]}'")
                tensor = np.load(file_name)
                print(f'{file_name}.shape: {tensor.shape}')
                shape = tensor.shape
                
                if json_data[key]=="row":
                    # weight slicing along row
                    
                    ######## orignal, look at later comment to understand the change
                    #row_split_num=int(shape[0]/world_size)
                    
                    row_split_num=int(shape[1]/world_size)

                    for i in range(world_size):
                        start_row = i * row_split_num

                        ######## original, look at later comment about transpose to understand
                        #end_row = (i + 1) * row_split_num if i != world_size - 1 else shape[0]

                        end_row = (i + 1) * row_split_num if i != world_size - 1 else shape[1]

                        ############### Original, we have to remember W2 will get transposed during F.linear
                        ########## so we need to row slice the 2nd dim, as inputs.dot(W2.T) for W2

                        #tensor_slice = tensor[start_row:end_row, :]
                        tensor_slice = tensor[:, start_row:end_row]

                        # creating the folder path for this slice (e.g., output_folders/slice_0)
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')

                        save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                        print(f'row weight: {slice_folder}/{os.path.basename(file_name)}: {tensor_slice.shape}')

                elif json_data[key] == "col":

                    # weight slicing along column
                    if len(shape)==2:

                        ################### original, look at the later comment about transpose 
                        #col_split_num = int(shape[1] / world_size)

                        col_split_num = int(shape[0] / world_size)

                        for i in range(world_size):
                            start_col = i * col_split_num

                            ############# original, look at the later comment about transpose to understand
                            #end_col = (i + 1) * col_split_num if i != world_size - 1 else shape[1]

                            end_col = (i + 1) * col_split_num if i != world_size - 1 else shape[0]
                            
                            ######### original, but we have to remember col slicing needs to be done along 1st dim,
                            ######## because, W1 will get input.dot(W1.T)

                            #tensor_slice = tensor[:, start_col:end_col]
                            tensor_slice = tensor[start_col:end_col,:]

                            # Define the folder for this slice
                            slice_folder = os.path.join(output_base_path, f'rank_{i}')
                            save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                            print(f'col weight: {slice_folder}/{os.path.basename(file_name)}: {tensor_slice.shape}')

                    else:
                        # bias slicing along col. so only 1 dimension
                        col_split_num = int(shape[0] / world_size)

                        for i in range(world_size):
                            start_col = i * col_split_num
                            end_col = (i + 1) * col_split_num if i != world_size - 1 else shape[0]
                            tensor_slice = tensor[start_col:end_col]

                            # Define the folder for this slice
                            slice_folder = os.path.join(output_base_path, f'rank_{i}')
                            save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                            print(f'col bias: {slice_folder}/{os.path.basename(file_name)}: {tensor_slice.shape}')

                elif json_data[key] == "divide":

                    row_split_num=int(shape[0]/world_size)
                    for i in range(world_size):
                        tensor_div = tensor/world_size
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')
                        save_tensor_slice(tensor_div, slice_folder, os.path.basename(file_name))
                        print(f'divide bias:{slice_folder}/{os.path.basename(file_name)}: {tensor_div.shape}')

                else:
                    for i in range(world_size):
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')
                        save_tensor_slice(tensor, slice_folder, os.path.basename(file_name))

                        print(f'same: {slice_folder}/{os.path.basename(file_name)}: {tensor.shape}')
                
                # remove file_name from file_list
                file_list.remove(file_name)

                ###### later added for deletion procedure
                os.remove(file_name)

    print(f'inside function file list: {len(file_list)}')

    return file_list


def download_opt_weights(model_name, path, world_size, rank_path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))

    ########## Edited
    file_list = list_files_in_folder(path)
    print(f'len:{len(file_list)}, reched here')
    #print(f'{file_list}\n\n')
    cur_dir = os.getcwd()
    file = 'flexgen_tp/mlp_parallel.json'
    mlp_json_path = os.path.join(cur_dir, file)
    print(mlp_json_path)
    
    if os.path.exists(mlp_json_path):
        with open(mlp_json_path, 'r') as file:
            #print('\n\ READING THE FILE \n')
            mlp_json = file.read()
            mlp_json_data = json.loads(mlp_json)
    else:
        print(f"File not found: {mlp_json_path}")
    ######### modification for flexgen original
    #world_size=2
    output_base_path = rank_path# "/home/aktarafder/ranks"
    

    #print(f'E reacher here mlp_json:{mlp_json_data}')

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    # slice tensors for mlp json
    file_list2 = slice_tensors(file_list, mlp_json_data, output_base_path, world_size, attn=False)
    
    # slice tensors for attention json
    print(f'\n\nfile_list2 len:{len(file_list2)}\n\n')

    file = 'flexgen_tp/mha_parallel.json'
    mha_json_path = os.path.join(cur_dir, file)

    with open(mha_json_path, 'r') as file:
        mha_json = file.read()
        mha_json_data = json.loads(mha_json)
    
    file_list3 = slice_tensors(file_list2, mha_json_data, output_base_path, world_size, attn=True)
    
    # putting the rest in all ranks
    
    print(f"\n\n file_list3 len:{len(file_list3)}\n")
    
    
    for file_name in file_list3.copy():
        print('called outside')
        tensor = np.load(file_name)
        print(f'{file_name} put in both ranks')
        for i in range(world_size):

            # creating the folder path for this slice (e.g., output_folders/slice_0)
            slice_folder = os.path.join(output_base_path, f'rank_{i}')
            save_tensor_slice(tensor, slice_folder, os.path.basename(file_name))
        
        file_list3.remove(file_name)

        ######### deletion,update it later need this to check if we want to download aagain
        #os.remove(file_name)

    assert len(file_list3) == 0, "all files have not been placed to rank folders! some error"
    

    '''
    for file_name in file_list:
        for key in mlp_data.keys():
            if file_name.endswith(key):
                print(f"File '{file_name}' matches key '{key}' with value '{mlp_data[key]}'")
                tensor = np.load(file_name)
                print(f'{file_name}.shape: {tensor.shape}')
                shape = tensor.shape

                if mlp_data[key]=="row":
                    row_split_num=int(shape[0]/world_size)

                    for i in range(world_size):
                        start_row = i * row_split_num
                        end_row = (i + 1) * row_split_num if i != world_size - 1 else shape[0]
                        tensor_slice = tensor[start_row:end_row, :]

                        # Define the folder for this slice (e.g., output_folders/slice_0)
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')

                        # Save the slice in the appropriate folder
                        save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                        print(f'{slice_folder}/{file_name}: {tensor_slice.shape}')

                elif mlp_data[key] == "col":
                    if len(shape)==2:
                        col_split_num = int(shape[1] / world_size)

                        for i in range(world_size):
                            start_col = i * col_split_num
                            end_col = (i + 1) * col_split_num if i != world_size - 1 else shape[1]

                            tensor_slice = tensor[:, start_col:end_col]

                            # Define the folder for this slice
                            slice_folder = os.path.join(output_base_path, f'rank_{i}')
                            save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                            print(f'{slice_folder}/{file_name}: {tensor_slice.shape}')

                    else:
                        # bias slicing along col. so only 1 dimension
                        col_split_num = int(shape[0] / world_size)

                        for i in range(world_size):
                            start_col = i * col_split_num
                            end_col = (i + 1) * col_split_num if i != world_size - 1 else shape[0]
                            tensor_slice = tensor[start_col:end_col]

                            # Define the folder for this slice
                            slice_folder = os.path.join(output_base_path, f'rank_{i}')
                            save_tensor_slice(tensor_slice, slice_folder, os.path.basename(file_name))
                            print(f'{slice_folder}/{file_name}: {tensor_slice.shape}')

                elif mlp_data[key] == "divide":

                    row_split_num=int(shape[0]/world_size)
                    for i in range(world_size):
                        tensor_div = tensor/world_size
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')
                        save_tensor_slice(tensor_div, slice_folder, os.path.basename(file_name))
                        print(f'{slice_folder}/{file_name}: {tensor_div.shape}')

                else:
                    for i in range(world_size):
                        slice_folder = os.path.join(output_base_path, f'rank_{i}')
                        save_tensor_slice(tensor, slice_folder, os.path.basename(file_name))

                        print(f'{slice_folder}/{file_name}: {tensor.shape}')
        
        # remove file_name from file_list
        '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()

    download_opt_weights(args.model, args.path)

import torch
import time
import datetime
import os


# Check if we are in a distributed environment
if 'WORLD_SIZE' in os.environ:
    # We are in a distributed environment
    local_rank = int(os.environ['LOCAL_RANK'])
    n_gpu = int(os.environ['WORLD_SIZE'])
else:
    # We are in a single machine environment
    local_rank = 0
    n_gpu = 1
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

# Setup distributed training
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(0, 18000))
_time = time.strftime("%y_%m_%d_%H:%M:%S", time.localtime())

assert local_rank == torch.distributed.get_rank(), \
    "local_rank {} is not equal to torch.distributed.get_rank() {}".format(local_rank, torch.distributed.get_rank())
assert n_gpu == torch.distributed.get_world_size(), \
    "n_gpu {} is not equal to torch.distributed.get_world_size() {}".format(n_gpu, torch.distributed.get_world_size())

print("local_rank: {}, n_gpu: {}".format(local_rank, n_gpu))
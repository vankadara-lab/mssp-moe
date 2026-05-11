"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from manager import MANAGER
from report import build_training_report
from scaling import classify_parameter

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_batches = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
only_logs = False # if True, skip checkpoint saving (only wandb logs, near-zero disk usage)
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
seed = 1337

# wandb logging
wandb_log = True # False # disabled by default
wandb_entity = 'mup_limitations'
wandb_project = 'moe-scaling'
wandb_run_name = 'nanomoe'

# data
dataset = 'dolma3_mix-150B-1025'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
sequence_length = 1024
data_seed = 42 # seed for data shuffling (separate from model init seed)

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attn_bias = False # separate bias toggle for attention c_attn and c_proj
weight_tying = False # tie wte and lm_head weights

# moe
# -- core
n_exp = 1 # if n_exp = 1 we just use regular MLP layers
top_k = 2
routing_fn = "softmax"  # "softmax" or "sigmoid" for expert weight computation
moe_implementation = "nanomoe"  # "nanomoe" (batched BMM) or "monitor" (per-expert modules)
mlp_ratio = 4  # dense MLP hidden dim = mlp_ratio * n_embd
moe_ratio = 4  # expert MLP hidden dim = moe_ratio * n_embd
# -- layer structure
stride = 2      # ratio of moe to dense layers (e.g. stride=2 means alternating dense and moe layers, stride=1 means full moe)
n_dense_layers_before_moe = 0
# -- auxiliary loss
use_aux_loss = False
aux_loss_weight = 0.01   # 0.01 as in https://arxiv.org/pdf/2101.03961
# -- router z-loss
use_router_z_loss = False
router_z_loss_weight = 0.001 # mentioned as default choice here: https://cameronrwolfe.substack.com/p/nano-moe
# -- noisy top-k
use_noisy_top_k = False
# -- token dropping
drop_tokens = False  # enforce expert capacity limit; False = no token dropping
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
# -- special init/routing
fixed_router = False  # freeze router weights (uniform routing)
tied_expert_init = False  # initialize all experts identically (copy expert 0)
# -- memory
gradient_checkpointing = False  # trade compute for memory by recomputing block activations

# scaling
# -- core
scaling = "standard"  # scaling config (see scaling.py)
moe_scaling_alpha = 0.0  # scale routing weights by (base_top_k / top_k)^alpha (0 = no-op)
base_top_k = 1  # reference top_k for moe_scaling_alpha
# -- forward-pass multipliers
input_forward_mult = 1.0
dense_hidden_forward_mult = 1.0
expert_hidden_forward_mult = 1.0
output_forward_mult = 1.0
# -- router scaling
router_forward_mult = 1.0         # forward-pass multiplier on router probs
# -- learning rate multipliers
attn_qkv_lr_mult = 1.0
attn_proj_lr_mult = 1.0
mlp_up_lr_mult = 1.0
mlp_down_lr_mult = 1.0
expert_up_lr_mult = 1.0
expert_down_lr_mult = 1.0
router_lr_mult = 1.0
input_lr_mult = 1.0
output_lr_mult = 1.0
# -- init multipliers (on init std)
attn_qkv_init_mult = 1.0
attn_proj_init_mult = 1.0
mlp_up_init_mult = 1.0
mlp_down_init_mult = 1.0
expert_up_init_mult = 1.0
expert_down_init_mult = 1.0
router_init_mult = 1.0
input_init_mult = 1.0
output_init_mult = 1.0

# optimizer
optimizer_type = "adamw"  # "adamw" or "sgd"
learning_rate = 6e-4 # max learning rate
max_steps = 600000 # total number of training steps (used for LR schedule)
stop_at = None # stop training at this step (None = use max_steps)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_steps = 2000 # how many steps to warm up for
min_lr = None # minimum learning rate, defaults to learning_rate/10 per Chinchilla

# torch module monitor settings (for coordinate checking and activation/gradient monitoring)
monitor = False
monitor_modules_regex = '|'.join([
    r'transformer\.wte',
    r'transformer\.h\.[0-5]\.ln_1',
    r'transformer\.h\.[23]\.attn',
    r'transformer\.h\.[23]\.ln_2',
    r'transformer\.h\.[2]\.mlp',
    r'transformer\.h\.3\.expert_module\.router\.w_g',
    r'transformer\.h\.3\.expert_module\.w_g',
    r'.*transformer\.h\.[0-7]\.expert_module\.experts\.expert[01](?!\d)',
    r'.*transformer\.h\.3\.expert_module\.experts\.expert\d(?!\d)',
    r'transformer\.ln_f',
    r'lm_head',
])
reference_model = False

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ('float32' if not torch.cuda.is_available() else 'float16')
compile = False # use PyTorch 2.0 to compile the model to be faster
compile_mode = "default" # torch.compile mode: "default", "reduce-overhead", "max-autotune"
num_threads = 0 # number of CPU threads (0 = use PyTorch default)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
if min_lr is None:
    min_lr = 0.1 * learning_rate
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# derive wandb_run_name from config file if not explicitly set
if wandb_run_name == 'nanomoe':
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            config_basename = os.path.splitext(os.path.basename(arg))[0]
            wandb_run_name = config_basename + ' ' + time.strftime('%Y-%m-%d %H:%M:%S')
            break
if num_threads > 0:
    torch.set_num_threads(num_threads)
print(config)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl' )
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_step = gradient_accumulation_steps * ddp_world_size * batch_size * sequence_length
print(f"tokens per step will be: {tokens_per_step:,}")

if master_process and not only_logs:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_script_dir, 'data', dataset)

class DataLoader:
    """Epoch-based sequential sampler: non-overlapping chunks, reshuffled each epoch."""
    def __init__(self, data_path, batch_size, sequence_length, rank=0, world_size=1, seed=42):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        # non-overlapping chunks: chunk i -> tokens [i*seq_len : (i+1)*seq_len]
        self.n_chunks = len(self.data) // sequence_length
        self.epoch = 0
        self.cursor = 0  # global position in permutation (shared across all ranks)
        self._shuffle()

    def _shuffle(self):
        rng = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
        self.perm = rng.permutation(self.n_chunks)

    def next_batch(self):
        B, W = self.batch_size, self.world_size
        start = self.cursor + self.rank * B
        if start + B > len(self.perm):
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
            start = self.cursor + self.rank * B
        chunk_ids = self.perm[start : start + B]
        self.cursor += B * W  # advance by total consumed across all ranks
        ix = chunk_ids * self.sequence_length
        x = np.stack([self.data[i : i + self.sequence_length] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.sequence_length] for i in ix])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    def reset(self):
        """Reset to start of current epoch (used for eval)."""
        self.cursor = 0

    def advance_to_step(self, step, gradient_accumulation_steps):
        """Fast-forward to the right position after resuming from checkpoint."""
        chunks_per_step = gradient_accumulation_steps * self.batch_size * self.world_size
        total = step * chunks_per_step
        self.epoch = total // self.n_chunks
        self.cursor = total % self.n_chunks
        self._shuffle()

def to_device(x, y):
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

train_loader = DataLoader(os.path.join(data_dir, 'train.bin'), batch_size, sequence_length,
                          rank=ddp_rank if ddp else 0, world_size=ddp_world_size, seed=data_seed)
val_loader = DataLoader(os.path.join(data_dir, 'val.bin'), batch_size, sequence_length,
                        rank=ddp_rank if ddp else 0, world_size=ddp_world_size, seed=data_seed + 1)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
step = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, sequence_length=sequence_length,
                  bias=bias, attn_bias=attn_bias, weight_tying=weight_tying, vocab_size=None, n_exp=n_exp, top_k=top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, drop_tokens=drop_tokens,
                  train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, stride=stride,
                  n_dense_layers_before_moe=n_dense_layers_before_moe,
                  scaling=scaling,
                  routing_fn=routing_fn,
                  moe_scaling_alpha=moe_scaling_alpha,
                  base_top_k=base_top_k,
                  moe_implementation=moe_implementation,
                  fixed_router=fixed_router,
                  mlp_ratio=mlp_ratio,
                  moe_ratio=moe_ratio,
                  gradient_checkpointing=gradient_checkpointing,
                  input_forward_mult=input_forward_mult,
                  dense_hidden_forward_mult=dense_hidden_forward_mult,
                  expert_hidden_forward_mult=expert_hidden_forward_mult,
                  output_forward_mult=output_forward_mult,
                  router_forward_mult=router_forward_mult,
                  attn_qkv_lr_mult=attn_qkv_lr_mult,
                  attn_proj_lr_mult=attn_proj_lr_mult,
                  mlp_up_lr_mult=mlp_up_lr_mult,
                  mlp_down_lr_mult=mlp_down_lr_mult,
                  expert_up_lr_mult=expert_up_lr_mult,
                  expert_down_lr_mult=expert_down_lr_mult,
                  router_lr_mult=router_lr_mult,
                  input_lr_mult=input_lr_mult,
                  output_lr_mult=output_lr_mult,
                  attn_qkv_init_mult=attn_qkv_init_mult,
                  attn_proj_init_mult=attn_proj_init_mult,
                  mlp_up_init_mult=mlp_up_init_mult,
                  mlp_down_init_mult=mlp_down_init_mult,
                  expert_up_init_mult=expert_up_init_mult,
                  expert_down_init_mult=expert_down_init_mult,
                  router_init_mult=router_init_mult,
                  input_init_mult=input_init_mult,
                  output_init_mult=output_init_mult,
                  tied_expert_init=tied_expert_init) # start with model_args from command line
print('\n\n')
print(model_args)
print('\n\n')
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    assert not only_logs, "Cannot resume training with only_logs=True (no checkpoint to resume from)"
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'sequence_length', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    step = checkpoint['step']
    best_val_loss = checkpoint['best_val_loss']
    train_loader.advance_to_step(step, gradient_accumulation_steps)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'sequence_length', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if sequence_length < model.config.sequence_length:
    model.crop_sequence_length(sequence_length)
    model_args['sequence_length'] = sequence_length # so that the checkpoint will have the right value
model.to(device)
n_moe_layers = sum(1 for i in range(n_layer) if i >= n_dense_layers_before_moe and (i % stride) == (stride - 1)) if n_exp > 1 else 0

training_monitor = None
_reference_model = None
coordinate_check = None
if monitor:
    from torch_module_monitor import ModuleMonitor, RefinedCoordinateCheck
    import logging
    logging.basicConfig(level=logging.INFO)

    if reference_model:
        print("Initializing reference model")
        _reference_model = GPT(gptconf)
        _reference_model.to(device)
        _reference_model.load_state_dict(model.state_dict())

    training_monitor = ModuleMonitor(
        monitor_step_fn=lambda step: step <= 20 or step % 20 == 0, # monitor first 20 steps, then every 20 steps after that
        included_modules_regex=monitor_modules_regex,
    )
    training_monitor.add_activation_metric("L2norm", lambda activations: torch.linalg.vector_norm(activations, ord=2, dim=-1))
    training_monitor.add_parameter_metric("L2norm", lambda parameters: torch.linalg.vector_norm(parameters.flatten(), ord=2))
    training_monitor.add_gradient_metric("L2norm", lambda gradients: torch.linalg.vector_norm(gradients.flatten(), ord=2))
    training_monitor.set_module(model)

    if reference_model:
        training_monitor.add_activation_difference_metric("L2norm", lambda activations, reference_activations: torch.linalg.vector_norm((activations - reference_activations), ord=2, dim=-1))
        training_monitor.add_parameter_difference_metric("L2norm", lambda parameters, reference_parameters: torch.linalg.vector_norm((parameters - reference_parameters).flatten(), ord=2))
        training_monitor.set_reference_module(_reference_model)
        coordinate_check = RefinedCoordinateCheck(training_monitor)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, eps=eps, optimizer_type=optimizer_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print(f"compiling the model with mode={compile_mode!r}... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, mode=compile_mode) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_train_loader = DataLoader(os.path.join(data_dir, 'train.bin'), batch_size, sequence_length,
                                   rank=ddp_rank if ddp else 0, world_size=ddp_world_size, seed=data_seed + 2)
    val_loader.reset()
    for split, loader in [('train', eval_train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = to_device(*loader.next_batch())
            with ctx:
                _, ce_loss, _, _, _ = model(X, Y)
            losses[k] = ce_loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return learning_rate * (it + 1) / (warmup_steps + 1)
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name, config={**config, **({"slurm_job_id": slurm_job_id} if slurm_job_id else {})})

# training loop
X, Y = to_device(*train_loader.next_batch()) # fetch the very first batch
t0 = time.time()
local_step = 0 # number of steps in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
if master_process:
    print(build_training_report(raw_model, config))
    # log per-parameter initialization stats and learning rates to wandb
    if wandb_log:
        init_log = {}
        for name, p in raw_model.named_parameters():
            role = classify_parameter(name)
            lr_s = raw_model.scaling.lr_scale(role, raw_model.config)
            lr_s *= raw_model._role_lr_mult(role)
            init_log[f"initialization/{name}_mean"] = p.mean().item()
            init_log[f"initialization/{name}_std"] = p.std().item()
            init_log[f"learning_rates/{name}"] = learning_rate * lr_s
        wandb.log(init_log, step=0)
while True:

    # determine and set the learning rate for this step
    lr = get_lr(step) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

    # evaluate the loss on train/val sets and write checkpoints
    if step % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "step": step,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/lr": lr,
            }, step=step)
        if not only_logs and (losses['val'] < best_val_loss or always_save_checkpoint):
            best_val_loss = losses['val']
            if step > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if step == 0 and eval_only:
        break

    if monitor:
        training_monitor.begin_step(step)

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    if n_exp > 1:
        MANAGER.reset_expert_stats()
        MANAGER.reset_output_l2()
        MANAGER.reset_router_weight_extremes()
        MANAGER.reset_sigmoid_ratio()
    # loss accumulators for logging (on device to avoid GPU sync per micro-batch)
    ce_loss_accum = torch.tensor(0.0, device=device)
    total_loss_accum = torch.tensor(0.0, device=device)
    aux_loss_accum = torch.tensor(0.0, device=device)
    z_loss_accum = torch.tensor(0.0, device=device)
    for micro_step in range(gradient_accumulation_steps):
        if monitor and reference_model:
            with torch.no_grad():
                with ctx:
                    _, _, _, _, _ = _reference_model(X, Y)
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, ce_loss, total_loss, aux_loss, z_loss = model(X, Y)
            total_loss = total_loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # accumulate losses for logging (detach to avoid graph retention, stay on device)
        ce_loss_accum += ce_loss.detach()
        total_loss_accum += total_loss.detach()
        if aux_loss is not None:
            aux_loss_accum += aux_loss.detach()
        if z_loss is not None:
            z_loss_accum += z_loss.detach()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = to_device(*train_loader.next_batch())
        # backward pass, with gradient scaling if training in fp16
        if monitor and gradient_checkpointing:
            with training_monitor.no_monitor():
                scaler.scale(total_loss).backward()
        else:
            scaler.scale(total_loss).backward()
        if device_type == 'cpu':
            print(f"  micro_step {micro_step+1}/{gradient_accumulation_steps}, loss {total_loss.item() * gradient_accumulation_steps:.4f}")
        if monitor and reference_model:
            with ctx:
                coordinate_check.refined_coordinate_check()
        if monitor:
            training_monitor.after_micro_batch()
    if monitor:
        training_monitor.monitor_gradients(before_clip=True)
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if monitor:
        training_monitor.monitor_parameters()
        training_monitor.monitor_gradients()
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if monitor:
        training_monitor.end_step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if master_process:
        # get loss as float, averaged over all micro-batches. note: this is a CPU-GPU sync point
        lossf = (ce_loss_accum / gradient_accumulation_steps).item()
        total_lossf = total_loss_accum.item()  # already averaged (sum of loss_i/N)
        # memory statistics
        if device_type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device)
            max_mem_allocated = torch.cuda.max_memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            max_mem_reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            mem_allocated = max_mem_allocated = mem_reserved = max_mem_reserved = 0
        loss_str = f"step {step}: loss {lossf:.4f}"
        if use_aux_loss or use_router_z_loss:
            loss_str += f", total_loss {total_lossf:.4f}"
        print(f"{loss_str}, time {dt*1000:.2f}ms, mem {mem_allocated/1e9:.2f}GB (peak {max_mem_allocated/1e9:.2f}GB)")
        if wandb_log:
            log_dict = {
                "step": step,
                "train/loss": lossf,
                "train/lr": lr,
                "train/tokens": (step + 1) * tokens_per_step,
                "train/tokens_per_sec": tokens_per_step / dt,
                "data/epoch": train_loader.epoch,
                "time_ms": dt*1000,
                "gpu_memory/allocated_GB": mem_allocated / 1e9,
                "gpu_memory/max_allocated_GB": max_mem_allocated / 1e9,
                "gpu_memory/reserved_GB": mem_reserved / 1e9,
                "gpu_memory/max_reserved_GB": max_mem_reserved / 1e9,
            }
            if grad_clip != 0.0:
                log_dict["train/grad_norm"] = grad_norm.item()
            if use_aux_loss or use_router_z_loss:
                log_dict["train/total_loss"] = total_lossf
            if use_aux_loss:
                log_dict["train/aux_loss"] = (aux_loss_accum / gradient_accumulation_steps).item()
            if use_router_z_loss:
                log_dict["train/z_loss"] = (z_loss_accum / gradient_accumulation_steps).item()
            if n_exp > 1:
                token_stats, weight_stats = MANAGER.aggregate_expert_stats(n_moe_layers)
                for layer_idx in range(len(token_stats)):
                    for exp_idx in range(len(token_stats[layer_idx])):
                        log_dict[f"moe_tokens/layer{layer_idx}/expert{exp_idx}"] = token_stats[layer_idx][exp_idx].item()
                        log_dict[f"moe_weights/layer{layer_idx}/expert{exp_idx}"] = weight_stats[layer_idx][exp_idx].item()
                    log_dict[f"moe_dropped/layer{layer_idx}"] = 1.0 - token_stats[layer_idx].sum().item() / top_k
                # output L2 norms
                l2_stats = MANAGER.aggregate_output_l2(n_moe_layers)
                for layer_idx, l2 in enumerate(l2_stats):
                    log_dict[f"moe_output_l2/layer{layer_idx}"] = l2.item()
                # router weight extremes
                w_maxes, w_mins = MANAGER.aggregate_router_weight_extremes(n_moe_layers)
                for layer_idx in range(len(w_maxes)):
                    log_dict[f"moe_router_weight_max/layer{layer_idx}"] = w_maxes[layer_idx].item()
                    log_dict[f"moe_router_weight_min/layer{layer_idx}"] = w_mins[layer_idx].item()
                # sigmoid ratios
                sig_ratios = MANAGER.aggregate_sigmoid_ratio(n_moe_layers)
                for layer_idx, ratio in enumerate(sig_ratios):
                    for exp_idx in range(len(ratio)):
                        log_dict[f"moe_sigmoid_ratio/layer{layer_idx}/expert{exp_idx}"] = ratio[exp_idx].item()
            if monitor:
                log_dict.update(training_monitor.get_step_metrics())
            wandb.log(log_dict, step=step)
    step += 1
    local_step += 1

    # termination conditions
    final_step = stop_at if stop_at is not None else max_steps
    if step > final_step:
        break

# final evaluation
if master_process:
    losses = estimate_loss()
    print(f"final step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if wandb_log:
        wandb.log({
            "step": step,
            "val/loss": losses['val'],
        }, step=step)

if ddp:
    destroy_process_group()

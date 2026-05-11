
# model
n_layer = 8
n_head = 16
n_embd = 1024
stride = 1 # full moe

# moe
moe_implementation = "nanomoe"
n_exp = 32
top_k = 16
use_aux_loss=True
moe_ratio = 0.5
routing_fn = "sigmoid"

# total batch size
sequence_length = 1024
batch_size = 1
gradient_accumulation_steps = 512

max_steps = 4768
warmup_steps = 700

# eval stuff
eval_interval = 1000
eval_batches = 200

# learning rate
learning_rate = 4.64159e-3

# scaling
scaling = "mup_allscale"
moe_scaling_alpha = 1.0
tied_expert_init = True

# tuned multipliers
input_lr_mult = 16
attn_qkv_lr_mult = 2
attn_proj_lr_mult = 2
expert_up_lr_mult = 2
expert_down_lr_mult = 128
router_lr_mult = 8
output_lr_mult = 8
expert_down_init_mult = 4

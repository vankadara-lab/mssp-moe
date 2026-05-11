
# model
n_layer = 4
n_head = 4
n_embd = 256
stride = 2  # blocks [dense, MoE, dense, MoE]

# moe (bottleneck: d_hidden = n_embd * moe_ratio = 32, 2x the base bottleneck=16
# width; n_exp halved so expert params per MoE layer stay constant at 524288.
# top_k holds the current top_k/n_exp = 1/2 active-fraction of the existing
# bottleneck/sparse config)
moe_implementation = "nanomoe"
n_exp = 32
top_k = 16
use_aux_loss = True
moe_ratio = 1/8
routing_fn = "sigmoid"

# total batch size
sequence_length = 1024
batch_size = 8
gradient_accumulation_steps = 64

max_steps = 4768
warmup_steps = 700

# eval stuff
eval_interval = 1000
eval_batches = 200

# learning rate
learning_rate = 7.74264e-3

# scaling
scaling = "mup_bottleneck"
moe_scaling_alpha = 1.0

# tuned multipliers
input_lr_mult = 32
attn_qkv_lr_mult = 4
attn_proj_lr_mult = 1
mlp_up_lr_mult = 2
mlp_down_lr_mult = 4
expert_up_lr_mult = 4
expert_down_lr_mult = 256
router_lr_mult = 0.5
output_lr_mult = 8
input_init_mult = 4
attn_qkv_init_mult = 0.125
attn_proj_init_mult = 0.25
mlp_up_init_mult = 0.5
mlp_down_init_mult = 2
expert_up_init_mult = 0.125
router_init_mult = 0.03125
output_init_mult = 0.25

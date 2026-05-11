
# model
n_layer = 8
n_head = 12
n_embd = 768
stride = 1 # full moe

# moe
moe_implementation = "nanomoe"
n_exp = 192
top_k = 96
use_aux_loss=True
moe_ratio = 1/48
routing_fn = "sigmoid"

# total batch size
sequence_length = 1024
batch_size = 2
gradient_accumulation_steps = 256

max_steps = 4768
warmup_steps = 700

# eval stuff
eval_interval = 1000
eval_batches = 200

# learning rate
learning_rate = 7.74e-3

# scaling
scaling = "mup_bottleneck"
moe_scaling_alpha = 1.0

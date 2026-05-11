
# model
n_layer = 8
n_head = 8
n_embd = 512
stride = 1 # full moe

# moe
moe_implementation = "nanomoe"
n_exp = 128
top_k = 64
use_aux_loss=True
moe_ratio = 1/32
routing_fn = "sigmoid"

# total batch size
sequence_length = 1024
batch_size = 4
gradient_accumulation_steps = 128

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

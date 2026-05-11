
# model
n_layer = 8
n_head = 16
n_embd = 1024
stride = 1 # full moe

# moe
moe_implementation = "softmoe"
n_exp = 32
top_k = 32
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
learning_rate = 7.74e-3

# scaling
scaling = "mup_allscale"
moe_scaling_alpha = 1.0
tied_expert_init = True

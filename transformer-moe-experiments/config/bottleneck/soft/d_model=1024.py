
# model
n_layer = 8
n_head = 16
n_embd = 1024
stride = 1 # full moe 

# moe
moe_implementation = "softmoe"
n_exp = 256
top_k = 256
moe_ratio = 1/64
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

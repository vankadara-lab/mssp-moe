
# model
n_layer = 8
n_head = 32
n_embd = 2048
stride = 1 # full moe 

# moe
moe_implementation = "softmoe"
n_exp = 512
top_k = 512
moe_ratio = 1/128
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
scaling = "mup_bottleneck"
moe_scaling_alpha = 1.0

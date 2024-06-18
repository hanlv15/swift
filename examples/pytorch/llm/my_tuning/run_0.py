import run_base

DEVICE = "0"

# Llama-3

# dora
# for lr in ["5e-5"]:
#     run_base.run_llama3_8b_dora(lr=lr, device=DEVICE)
    
# lora
for lr in ["1e-4"]:
    run_base.run_llama3_8b_lora(lr=lr, device=DEVICE)

# rslora


# Phi-3-small-8k-instruct
# lora
# for lr in ["3.5e-4"]:
#     run_base.run_phi3_small_lora(lr=lr, device=DEVICE)
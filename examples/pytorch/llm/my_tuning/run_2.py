import run_base

DEVICE = "2"

# Llama-3

# dora
for lr in ["1.7e-4", "8e-5", "1.4e-4", "1.6e-4"]:
    run_base.run_llama3_8b_dora(lr=lr, device=DEVICE)

# lora
for lr in ["1.5e-4", "2e-4"]:
    run_base.run_llama3_8b_lora(lr=lr, device=DEVICE)

# rslora

# Phi-3-small-8k-instruct
# lora
# for lr in ["3.5e-4"]:
#     run_base.run_phi3_small_lora(lr=lr, device=DEVICE)
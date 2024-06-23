from run_base import SFTModels, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora

DEVICE = "2"

# Llama-3
for lr in ["1.5e-4", "2e-4"]:
    run_dora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

for lr in ["1.5e-4", "2e-4"]:
    run_lora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# "5.1e-2", "5.2e-2", "2.6e-2", "2.7e-2", "2.8e-2", "2.9e-2"
# for lr in ["5.1e-2", "5.2e-2", "2.6e-2", "2.7e-2", "2.8e-2", "2.9e-2"]:
#     run_vera(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["5e-5", "1e-4"]:
#     run_lora_plus(SFTModels.llama_3_8b_instruct, lr, DEVICE)

for lr in ["1.5e-4", "1.7e-4", "1.9e-4", "2e-4"]:
    run_rslora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["7.5e-5"]:
#     run_pissa(SFTModels.llama_3_8b_instruct, lr, DEVICE)



# openchat-3.5
# for lr in ["9e-5", "2e-4", "1e-4"]:
#     run_dora(SFTModels.openchat_35, lr, DEVICE)


# for lr in ["9e-5", "2e-4"]:
#     run_lora(SFTModels.openchat_35, lr, DEVICE)


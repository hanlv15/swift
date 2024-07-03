from run_base import SFTModels, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora

DEVICE = "0"

# phi-3-medium
# for lr in ["8e-5", "1.2e-4", "1.4e-4"]:
#     run_lora(SFTModels.phi_3_medium_instruct, lr, DEVICE)

# Llama-3
# for lr in ["8e-5", "1.1e-4"]: # 8e-5
#     run_dora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

for lr in ["8e-5", "9e-5"]:
    run_lora(SFTModels.llama_3_8b_instruct, lr, DEVICE)


# for lr in ["3.2e-2", "3.4e-2"]:
#     run_vera(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# # for lr in ["9e-6", "2e-5"]:
# #     run_lora_plus(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["3e-5", "4e-5", "5e-5", "6e-5"]:
#     run_rslora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["8e-5"]:
#     run_pissa(SFTModels.llama_3_8b_instruct, lr, DEVICE)


# openchat-3.5
# for lr in ["1e-4", "1.3e-4"]:
#     run_lora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["1e-4", "1.3e-4"]:
#     run_dora(SFTModels.openchat_35, lr, DEVICE)
  

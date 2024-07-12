from run_base import SFTModels, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora

DEVICE = "1"

# Llama-3
# for lr in []:
#     run_dora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["2e-4"]:
#     run_lora(SFTModels.llama_3_8b_instruct, lr, DEVICE, data_version="3")

# for lr in ["5e-5", "7e-5", ]:
#     run_rslora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["5e-6", ]:
#     run_lora_plus(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["4.6e-2", "4.7e-2"]:
#     run_vera(SFTModels.llama_3_8b_instruct, lr, DEVICE)


# openchat-3.5
# for lr in ["8e-5"]:
#     run_dora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["3e-5"]: # "1.5e-4", "7e-5"
#     run_lora(SFTModels.openchat_35, lr, DEVICE)

for lr in ["7e-5"]:
    run_rslora(SFTModels.openchat_35, lr, DEVICE)

for lr in ["1e-5"]:
    run_lora_plus(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["3e-2"]:
#     run_vera(SFTModels.openchat_35, lr, DEVICE)

# phi-3-medium
# for lr in ["1.5e-4"]:
#     run_lora(SFTModels.phi_3_medium_instruct, lr, DEVICE)


# mistral-7b-instruct
# for lr in ["1e-4"]:
#     run_lora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["8.3e-5"]: # 8e-5
#     run_dora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["6e-6"]:
#     run_lora_plus(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["1.6e-5",]:
#     run_rslora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["3e-2"]:
#     run_vera(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)




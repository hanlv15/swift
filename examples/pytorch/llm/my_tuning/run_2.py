from run_base import SFTModels, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora

DEVICE = "2"

# Llama-3
# for lr in []:
#     run_dora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["1.1e-4"]:
#     run_lora(SFTModels.llama_3_8b_instruct, lr, DEVICE, data_version="3")

# for lr in ["9e-5", "1e-4"]:
#     run_rslora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["7e-6"]:
#     run_lora_plus(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["4.8e-2", "4.9e-2"]:
#     run_vera(SFTModels.llama_3_8b_instruct, lr, DEVICE)


# openchat-3.5
# for lr in ["1e-4"]:
#     run_dora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["4e-5"]:
#     run_lora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["6.5e-5"]:
#     run_rslora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["7.5e-6"]:
#     run_lora_plus(SFTModels.openchat_35, lr, DEVICE)

for lr in ["9e-3"]:
    run_vera(SFTModels.openchat_35, lr, DEVICE)

# mistral-7b-instruct
# for lr in ["1.5e-4"]: # 1.5e-4
#     run_lora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["1.15e-4"]:
#     run_dora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["9.5e-6", "4.5e-6"]:
#     run_lora_plus(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["1.8e-5"]:
#     run_rslora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

for lr in ["4.5e-2", "8e-3"]:
    run_vera(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)


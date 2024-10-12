from run_base import SFTModels, DatasetName, run_lora, run_vera, run_dora, run_pissa, run_lora_plus, run_rslora, run_dora_with_info_or_not
import time
# time.sleep(3600*2) # 让程序等待2小时


DEVICE = "0"
# Llama-3
# for lr in ["7e-5", "9e-5"]:
#     for rank in ["2", "4"]:
#         run_dora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, rank=rank, data_version="1")

# for lr in ["9e-5"]:
#     for rank in ["16", "32"]:
#         run_dora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, rank=rank, data_version="1")


# for lr in ["1.8e-4"]:
#     with_info = True
#     run_dora_with_info_or_not(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, with_info, data_version="2")

############################
# for lr in ["1e-4"]: #  covmis: 9.5e-5 1.05e-4
#     run_dora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, data_version="1")

for lr in ["1e-4", "8e-5", "7e-5", ]:
    run_dora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, data_version="1")

# for lr in ["1e-4"]:
#     run_lora(SFTModels.llama_3_8b_instruct, lr, DatasetName.covmis, DEVICE, data_version="1")

# for lr in ["1e-5", "3e-5", ]:
#     run_rslora(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["3e-6", ]:
#     run_lora_plus(SFTModels.llama_3_8b_instruct, lr, DEVICE)

# for lr in ["3.2e-2", "3.4e-2"]:
#     run_vera(SFTModels.llama_3_8b_instruct, lr, DEVICE)


# openchat-3.5
# for lr in ["6e-5"]: # 
#     run_dora(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["7.5e-5"]:
#     run_lora(SFTModels.openchat_35, lr, DEVICE)

# "9e-6" "2e-5"
# for lr in ["3.5e-5"]:
#     run_rslora(SFTModels.openchat_35, lr, DEVICE)

# "5.5e-6", "6.5e-6", "7.5e-6",
# for lr in ["5.5e-6"]:
#     run_lora_plus(SFTModels.openchat_35, lr, DEVICE)

# for lr in ["1.9e-2"]:
#     run_vera(SFTModels.openchat_35, lr, DEVICE)

# mistral-7b-instruct
# for lr in ["7e-5"]:
#     run_lora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# for lr in ["6.5e-5"]:
#     run_dora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# # "5.5e-6", "9.5e-6"
# for lr in ["5.5e-6", "6.5e-6"]:
#     run_lora_plus(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)


# for lr in ["1.4e-5",]:
#     run_rslora(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# "3.1e-2", "3.3e-2", "3.5e-2", "3.7e-2", "3.9e-2" "4.1e-2"
# "2.3e-2", "2.5e-2", "2.7e-2" 

# for lr in ["5e-3", ]:
#     run_vera(SFTModels.mistral_7b_instruct_v3, lr, DEVICE)

# phi-3-medium
# for lr in ["8e-5", "1.2e-4", "1.4e-4"]:
#     run_lora(SFTModels.phi_3_medium_instruct, lr, DEVICE)

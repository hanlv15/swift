import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]
lrs1 = ["2e-5", "3e-5", "4e-5", "5e-5", "6e-5", "7e-5", "8e-5", "9e-5", "1e-4"]
lrs2 = ["1e-4", "1.5e-4", "2e-4", "1.1e-4", "1.2e-4", "1.3e-4", "1.4e-4", "1.6e-4", "1.7e-4", "1.8e-4", "1.9e-4"]

lr_del = ["1.1e-4", "1.5e-4"]

lrs = [value for value in lrs2 if value not in lr_del]

# openchat 3.5
# for sft_type in ["adalora"]:
#     for lr in ["3.7e-4", "3.9e-4", "3.2e-4", "3.4e-4", "3.8e-4", "4e-4"]:
#         for j in ["with_solar_info/brave"]:
#             data_version = "1"

#             # if j == "without_info":
#             #     for i in test_sizes:
#             #         subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", f"{i}", "1", j, data_version])

#             # sft
#             for i in [1.0]:
#                 subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft2.sh", "0.2", f"{i}", sft_type, "3", lr, j, data_version])

# lora+
# for lr in ["2.4e-5", "2.6e-5", "2.8e-5", "3.2e-5", "3.4e-5", "2.2e-5"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/lora_plus.sh", "0.2", "1.0", "lora", rank, lr, "with_solar_info/brave", data_version])

# galore
# for lr in ["1e-5"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/openchat_3.5/galore/sft.sh", "0.2", "1.0", rank, lr, "with_solar_info/brave", data_version])

# dora
for lr in ["3.6e-5", "3.8e-5"]:
    data_version = "1"
    rank = "8"
    subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/dora.sh", "0.2", "1.0", "lora", rank, lr, "with_solar_info/brave", data_version])

# rslora
# for lr in ["1.7e-4", "1.8e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/rslora.sh", "0.2", "1.0", "lora", rank, lr, "with_solar_info/brave", data_version])

# qlora
# for lr in ["1.4e-4", "1.7e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/qlora.sh", "0.2", "1.0", "lora", rank, lr, "with_solar_info/brave", data_version])


# fusechat-7b-varm
# lora+
# for lr in ["1.9e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/fusechat-7b-varm/lora/lora_plus.sh", "0.2", "1.0", "lora", rank, lr, data_version])



# gemma-7b-it
# lora+
# for lr in ["5e-5", "1e-4", "1.5e-4", "2e-4", "1.3e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/gemma-7b-it/lora/lora_plus.sh", "0.2", "1.0", "lora", rank, lr, data_version])

# merlinite-7b
# lora+
# for lr in ["1.9e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/merlinite-7b/lora/lora_plus.sh", "0.2", "1.0", "lora", rank, lr, data_version])

# lora
# for lr in ["1.2e-4", "1.6e-4"]:
#     data_version = "1"
#     rank = "3"
#     subprocess.run(["bash", f"my_tuning/merlinite-7b/lora/lora.sh", "0.2", "1.0", "lora", rank, lr, data_version])


# Mistral-7B-Instruct-v0.2
# for sft_type in ["adalora"]:
#     for lr in ["1e-3", "1.1e-3"]:
#         data_version = "1"
#         for i in [1.0]:
#             subprocess.run(["bash", f"my_tuning/Mistral-7B-Instruct-v0.2/lora/sft.sh", "0.2", f"{i}", sft_type, lr, data_version])


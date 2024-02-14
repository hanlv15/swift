import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]
lrs1 = ["2e-5", "3e-5", "4e-5", "5e-5", "6e-5", "7e-5", "8e-5", "9e-5", "1e-4"]
lrs2 = ["1e-4", "1.5e-4", "2e-4", "1.1e-4", "1.2e-4", "1.3e-4", "1.4e-4", "1.6e-4", "1.7e-4", "1.8e-4", "1.9e-4"]

lr_del = ["1.1e-4", "1.5e-4"]

lrs = [value for value in lrs2 if value not in lr_del]


# openchat 3.5
for sft_type in ["adalora"]:
    for lr in ["3e-4", "3.3e-4"]:
        for j in ["with_solar_info/brave"]:
            data_version = "1"

            # if j == "without_info":
            #     for i in test_sizes:
            #         subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", f"{i}", "1", j, data_version])

            for i in [1.0]:
                subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", "0.2", f"{i}", sft_type, lr, j, data_version])


# neural-chat-v3-3
# for lr in ["1.4e-4", "1.6e-4", "1.7e-4", "1.8e-4", "1.9e-4"]:
#     data_version = "1"
#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/neural-chat-7b-v3-3/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])

# neural-chat-v3-3-Slerp
# for lr in ["2e-5", "1.1e-4"]:
#     data_version = "3.3"

#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/neural-chat-7b-v3-3-Slerp/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])


# marcoroni-7b
# for lr in ["2e-5", "8e-5", "9e-5", "1e-4", "1.1e-4"]:
#     data_version = "3.3"

#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/marcoroni-7b-v3/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])

# Mistral-7B-Instruct-v0.2
# for lr in ["6e-5", "7e-5", "8e-5", "9e-5", "1e-4"]:
#     data_version = "1"
#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/Mistral-7B-Instruct-v0.2/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])

# DPOpenHermes-7B-v2
# for lr in lrs:
#     data_version = "3.3"
#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/DPOpenHermes-7B-v2/lora/sft.sh", "0.2", f"{i}", lr, data_version])

# DareBeagle-7B-v2
# for lr in ["1.9e-4", "1e-4", "1.3e-4", "1.4e-4", "2e-4"]:
#     data_version = "1"
#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/DareBeagle-7B-v2/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])

# Turdus
# for lr in lrs1:
#     data_version = "1"
#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/Turdus/lora/sft.sh", "0.2", f"{i}", "lora", lr, data_version])

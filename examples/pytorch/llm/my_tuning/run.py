import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]
lrs = ["2e-5", "3e-5", "4e-5", "5e-5", "6e-5", "7e-5", "8e-5", "9e-5", "1e-4", "1.1e-4"]

# openchat 3.5
# for lr in ["7e-5"]:
#     for j in ["with_solar_info"]:
#         # data_version = ""
#         if j == "with_solar_info":
#             data_version = "3.3"
#         else:
#             data_version = "3"

#         # if j == "without_info":
#         #     for i in test_sizes:
#         #         subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", f"{i}", "1", j, data_version])

#         for i in [0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
#             subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", "0.2", f"{i}", lr, j, data_version])

# neural-chat-v3-3
# for lr in ["2e-5", "3e-5", "5e-5", "7e-5", "8e-5", "1.1e-4"]:# "4e-5", "6e-5", "9e-5", "1e-4"]:
#     data_version = "3.3"

#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/neural-chat-7b-v3-3/sft.sh", "0.2", f"{i}", lr, data_version])

# neural-chat-v3-3-Slerp
# for lr in ["2e-5", "1.1e-4"]:
#     data_version = "3.3"

#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/neural-chat-7b-v3-3-Slerp/sft.sh", "0.2", f"{i}", lr, data_version])


# marcoroni-7b
# for lr in ["2e-5", "8e-5", "9e-5", "1e-4", "1.1e-4"]:
#     data_version = "3.3"

#     for i in [1.0]:
#         subprocess.run(["bash", f"my_tuning/marcoroni-7b-v3/sft.sh", "0.2", f"{i}", lr, data_version])

# Mistral-7B-Instruct-v0.2
for lr in lrs:
    data_version = "3.3"
    for i in [1.0]:
        subprocess.run(["bash", f"my_tuning/Mistral-7B-Instruct-v0.2/sft.sh", "0.2", f"{i}", lr, data_version])


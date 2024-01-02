import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]

# ["8.5e-5", "5.5e-5"]
for lr in ["6e-5"]:
    for j in ["without_info"]:
        # data_version = ""
        if j == "with_solar_info":
            data_version = "3.3"
        else:
            data_version = "3"

        # if j == "without_info":
        #     for i in test_sizes:
        #         subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", f"{i}", "1", j, data_version])

        for i in [0.8, 0.9]:
            subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", "0.2", f"{i}", lr, j, data_version])


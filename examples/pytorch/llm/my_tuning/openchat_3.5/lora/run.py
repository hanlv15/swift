import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]

for j in ["with_info", "without_info", "with_solar_info"]:
    data_version = ""
    if j == "with_info":
        data_version = "2"
    elif j == "with_solar_info":
        data_version = "2.1"
    
    if j == "without_info":
        for i in test_sizes:
            subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", f"{i}", "1", j, data_version])

    for i in train_ratios:
        subprocess.run(["bash", f"my_tuning/openchat_3.5/lora/sft.sh", "0.2", f"{i}", j, data_version])


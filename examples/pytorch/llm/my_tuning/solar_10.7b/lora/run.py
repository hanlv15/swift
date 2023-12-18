import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]

for i in train_ratios:
    subprocess.run(["bash", f"my_tuning/orca2_7b/lora/sft.sh", f"{i}"])


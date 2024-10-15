import subprocess

train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]
lrs1 = ["2e-5", "3e-5", "4e-5", "5e-5", "6e-5", "7e-5", "8e-5", "9e-5", "1e-4"]
lrs2 = ["1e-4", "1.5e-4", "2e-4", "1.1e-4", "1.2e-4", "1.3e-4", "1.4e-4", "1.6e-4", "1.7e-4", "1.8e-4", "1.9e-4"]

lr_del = ["1.1e-4", "1.5e-4"]

lrs = [value for value in lrs2 if value not in lr_del]

data_version = "1"

class SFTModels:
	llama_3_8b_instruct = "Meta-Llama-3-8B-Instruct"
	openchat_35 = "openchat_3.5"
	phi_3_medium_instruct = "Phi-3-medium-4k-instruct"
	mistral_7b_instruct_v3 = "Mistral-7B-Instruct-v0.3"
	gemma_2_9b_it = "gemma-2-9b-it"

class DatasetName:
	covmis = "covmis"
	liar2 = "liar2"
	covmis_wsc = "covmis_wsc"


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

def run_lora(sft_model, lr, dataset_name, device, rank="8", rag_model="llama3", data_version=data_version, num_epochs="1"):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/lora.sh", dataset_name,
			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, num_epochs, device])

def run_pissa(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/pissa.sh", 
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_vera(sft_model, lr, device, vera_rank="256", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/vera.sh", 
   			"0.2", "1.0", "vera", vera_rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_rslora(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/rslora.sh", 
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_lora_plus(sft_model, lr, device, rank="8", rag_model="llama3", data_version=data_version):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/lora_plus.sh",
			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, device])

def run_dora(sft_model, lr, dataset_name, device, rank="8", rag_model="llama3", data_version=data_version, num_epochs="1"):
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/dora.sh", dataset_name,
   			"0.2", "1.0", "lora", rank, lr, f"with_{rag_model}_info/brave", data_version, num_epochs, device])

def run_dora_with_info_or_not(sft_model, lr, dataset_name, device, with_info, rank="8", data_version=data_version, num_epochs="1"):
	if with_info:
		with_info_or_not = "with_info"
	else:
		with_info_or_not = "without_info"
	subprocess.run(
		["bash", f"my_tuning/{sft_model}/lora/dora.sh", dataset_name,
   			"0.2", "1.0", "lora", rank, lr, with_info_or_not, data_version, num_epochs, device])

# Llama-3
# qlora
# for lr in ["1.5e-4"]:
#     data_version = "1"
#     rank = "2"
#     model_name = "mixtral"
#     subprocess.run(["bash", f"my_tuning/Meta-Llama-3-8B-Instruct/lora/qlora.sh", "0.2", "1.0", "lora", rank, lr, f"with_{model_name}_info/brave", data_version])

# boft
# def run_llama3_8b_boft(lr, device, block_size="8", n_butterfly_factor="2", model_name="llama3", data_version=data_version):
# 	subprocess.run(
# 		["bash", f"my_tuning/Meta-Llama-3-8B-Instruct/lora/boft.sh", 
#    			"0.2", "1.0", "boft", block_size, n_butterfly_factor, lr, f"with_{model_name}_info/brave", data_version, device])

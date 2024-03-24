from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import jsonlines, json
import os
from tqdm import tqdm

def cal_metric_single_llm(get_engine_config_request, inference, sft_args, ckpt_dir, train_loss, save=True):
    def load_metrics(file_dir, model_name, template_type):
        os.makedirs(file_dir, exist_ok=True)
        file_path = file_dir + '/' + f"{model_name}({template_type}).json"

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump([], f, indent=4)
        with open(file_path, "r") as f:
            metrics:list = json.load(f)
        return metrics
    
    def save_metrics(file_dir, model_name, template_type, metrics, save):
        if save:
            file_path = file_dir + '/' + f"{model_name}({template_type}).json"
            with open(file_path, "w") as f:
                json.dump(metrics, f, indent=4)

    def update_metrics(metrics, model_name, split_type, train_ratio, labels, preds, lr=None):
        new_item = {
            "model": model_name,
            "train_test_split": split_type,
            "train_ratio": train_ratio,
            "train_loss": train_loss,
        }
        
        if lr is not None:
            new_item.update(lr=lr)
        new_item.update(
            ACC=accuracy_score(labels, preds),
            F1=f1_score(labels, preds, average='macro'),
            Precision=precision_score(labels, preds, average='macro'),
            Recall=recall_score(labels, preds, average='macro'),
        )
        metrics.append(new_item)
        return new_item
    
    def get_with_or_without_info(train_dataset_path: str):
        with_or_without_info_list = ["with_info", "with_solar_info", "without_info"]
        search_engines = ["brave"]
        for with_or_without_info in with_or_without_info_list:
            if with_or_without_info == "with_solar_info":
                for search_engine in search_engines:
                    info_search = with_or_without_info + '/' + search_engine
                    if info_search in train_dataset_path:
                        return info_search
            else:
                if with_or_without_info in train_dataset_path:
                    return with_or_without_info
        raise Exception(f"with_or_without_info选择范围：{with_or_without_info_list}\nsearch_engines的选择范围：{search_engines}")
    
    def get_split_type(train_dataset_path: str):
        split_types = ["5:5", "6:4", "7:3", "8:2", "9:1"]
        for split_type in split_types:
            if split_type in train_dataset_path:
                return split_type
        raise Exception(f"split_type选择范围：{split_types}")
    
    def get_train_ratio(train_dataset_path: str):
        pos_train = train_dataset_path.find("/train_data_")
        if pos_train == -1:
            raise Exception("无法从训练数据路径中找到'/train_data_'")
        
        pos_end = train_dataset_path.find(".json", pos_train)
        return train_dataset_path[pos_train + len("/train_data_"): pos_end]
    
    def get_data_version(custom_train_dataset_path: str):
        pos_sub = custom_train_dataset_path.find("subtrain_data")
        if pos_sub == -1:
            raise Exception("无法从训练数据路径中找到'subtrain_data'")
        
        pos_end = custom_train_dataset_path.find("/", pos_sub)
        return custom_train_dataset_path[pos_sub + len("subtrain_data"): pos_end]
    
    def get_model_name(sft_args):
        model_id_or_path = sft_args.get("model_cache_dir", None)
        if model_id_or_path is None:
            model_id_or_path = sft_args["model_id_or_path"]
        return model_id_or_path[model_id_or_path.rfind('/') + 1:]
    
    def get_sft_type(sft_args):
        sft_type = sft_args["sft_type"]
        if sft_type == "lora":
            quantization_bit = sft_args["quantization_bit"]
            if sft_args["use_dora"]:
                sft_type = "dora"
            elif sft_args["use_rslora"]:
                sft_type = "rslora"
            elif sft_args["lora_lr_ratio"] is not None:
                sft_type += "_plus"
            elif quantization_bit != 0:
                sft_type = f"qlora-int{quantization_bit}"
        return sft_type
    
    def get_label(response):
        # 0:false, 2:true

        if response == "TRUE.":
            return 2
        elif response == "FALSE.":
            return 0
        else:
            raise Exception("Response既不是'TRUE.'，也不是'FALSE.'。")
        
    def get_lr(output_dir: str):
        pos_lr = output_dir.find('lr=')
        if pos_lr == -1:
            raise Exception("无法从训练数据路径中找到'lr'")
        pos_end = output_dir.find("-20", pos_lr)
        return output_dir[pos_lr + len("lr="): pos_end]

    wrong_ans = []

    # if train_ratio != 1:
    #     train_ratio = float(train_ratio)

    with_or_without_info = get_with_or_without_info(sft_args["custom_train_dataset_path"][0])
    split_type = get_split_type(sft_args["custom_train_dataset_path"][0])
    train_ratio = get_train_ratio(sft_args["custom_train_dataset_path"][0])
    data_version = get_data_version(sft_args["custom_train_dataset_path"][0])
    model_name = get_model_name(sft_args)
    template_type = sft_args["template_type"]
    sft_type = get_sft_type(sft_args)
    lora_rank = sft_args["lora_rank"]
    
    lr = get_lr(sft_args["output_dir"])

    # 判断metric是否已经存在，那么不用再计算
    exist = False
    if train_ratio == "1.0":
        file_dir = f"test_metric_single_llm/{with_or_without_info}/\
data{data_version}-split={split_type}-ratio={train_ratio}/{sft_type}"
        if sft_type == "adalora":
            r1, r2 = sft_args["adalora_target_r"], sft_args["adalora_init_r"]
            file_dir += f"-r={r1}_{r2}"
        else:
            file_dir += f"-r={lora_rank}" 
        metrics = load_metrics(file_dir, model_name, template_type)
        for item in metrics:
            if item["train_test_split"] == split_type and \
                item["train_ratio"] == train_ratio and item["lr"] == lr:

                # update_item_metric(item)
                exist = True
                break
    else:
        file_dir = f"test_metric_single_llm/{with_or_without_info}/\
data{data_version}-split={split_type}-sft={sft_type}-lr={lr}"
        metrics = load_metrics(file_dir, model_name, template_type)
        for item in metrics:
            if item["train_test_split"] == split_type and \
                item["train_ratio"] == train_ratio:

                # update_item_metric(item)
                exist = True
                break
    if exist:
        return wrong_ans

    data = []
    with jsonlines.open(
        f"../my_data/{with_or_without_info}/train_test_split/{split_type}/\
test_data{data_version}.jsonl", 'r') as f:
        for item in f.iter():
            data.append(item)

    # 推理
    vllm_engine, template, generation_config, lora_request = get_engine_config_request(ckpt_dir)
    print(f'{model_name} sft_type={sft_type} lr={lr} ')

    request_list = [{'query': i["query"]} for i in data]
    response_list = inference(
        vllm_engine, template, request_list, 
        generation_config=generation_config, 
        lora_request=lora_request,
        use_tqdm=True,
    )
    
    labels, preds = [], []
    for item_data, item_resp in zip(data, response_list):
        pred = get_label(item_resp["response"])
        label = get_label(item_data["response"])

        labels.append(label)
        preds.append(pred)

    # ratio为1.0
    if train_ratio == "1.0":
        new_metric = update_metrics(
            metrics, model_name, split_type, train_ratio, 
            labels=labels, preds=preds, lr=lr
        )
        metrics.sort(key=lambda x: (x["train_test_split"], x["train_ratio"], float(x["lr"])))
        save_metrics(file_dir, model_name, template_type, metrics, save=save)
    else:
        # 不同的ratio
        new_metric = update_metrics(
            metrics, model_name, split_type, train_ratio,
            labels=labels, preds=preds, 
        )
        metrics.sort(key=lambda x: (x["train_test_split"], x["train_ratio"]))
        save_metrics(file_dir, model_name, template_type, metrics, save=save)
    
    print(json.dumps(new_metric, indent=4))
    
    # with open("wrong_ans.txt", "a") as f:
    #     title = f'{model_name}, sft_type={sft_type}, lr={lr}, wrong={cnt["wrong"]}'
    #     f.writelines('\n'.join([title] + wrong_ans) + '\n\n')


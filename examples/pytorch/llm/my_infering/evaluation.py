from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import jsonlines, json
import os
from IPython import display
from tqdm import tqdm

def cal_metric_single_llm(sft_args, get_response, save=True, use_tqdm=False):
    cnt = {}
    cnt[0] = cnt[2] = cnt[-2] = cnt["wrong"] = 0
    
    data = []
    preds = []
    labels = []

    def print_metrics(labels, preds):
        print(f"ACC: {accuracy_score(labels, preds)}")
        print(f"F1: {f1_score(labels, preds, average='macro')}")
        print(f"Precision: {precision_score(labels, preds, average='macro')}")
        print(f"Recall: {recall_score(labels, preds, average='macro')}")
        print()

    def update_item_metric(item):
        item["ACC"] = accuracy_score(labels, preds)
        item["F1"] = f1_score(labels, preds, average='macro')
        item["Precision"] = precision_score(labels, preds, average='macro')
        item["Recall"] = recall_score(labels, preds, average='macro')

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

    def update_metrics(metrics, model_name, split_type, train_ratio, lr=None):
        new_item = {
            "model": model_name,
            "train_test_split": split_type,
            "train_ratio": train_ratio
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
    
    def get_model_name(model_cache_dir: str):
        return model_cache_dir[model_cache_dir.rfind('/') + 1:]

    def get_label(prompt, use_tqdm=False):
        # 0:false, 2:true
        response = get_response(prompt)[0].strip()
        if not use_tqdm:
            print(f"\nPrompt:\n{prompt}\nAnswer:\n{response}\n")

        if response == "TRUE.":
            return 2
        elif response == "FALSE.":
            return 0
        else:
            raise Exception("LLM的回答既不是'TRUE.'，也不是'FALSE.'。")
        
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
    model_name = get_model_name(sft_args["model_cache_dir"])
    template_type = sft_args["template_type"]
    sft_type = sft_args["sft_type"]

    lr = get_lr(sft_args["output_dir"])

    with jsonlines.open(
        f"../my_data/{with_or_without_info}/train_test_split/{split_type}/\
test_data{data_version}.jsonl", 'r') as f:
        
        for item in f.iter():
            data.append(item)
        
    data_iter = enumerate(data) if not use_tqdm else tqdm(enumerate(data), total=len(data))
    for i, item in data_iter:
        if not use_tqdm:
            display.clear_output(wait=True)
            print(f"{i + 1} / {len(data)}")
            print(cnt)
            if i > 0:
                print_metrics(labels, preds)

        pred = get_label(item["query"], use_tqdm=use_tqdm)

        if item["response"] == "FALSE.":
            label = 0
        elif item["response"] == "TRUE.":
            label = 2
        else:
            raise Exception()

        labels.append(label)
        preds.append(pred)

        if label != pred:
            wrong_ans.append(item["query"])
            cnt["wrong"] += 1
        cnt[pred] += 1
    
    if not use_tqdm:
        print_metrics(labels, preds)

    # ratio为1.0
    if train_ratio == "1.0":
        file_dir = f"test_metric_single_llm/{with_or_without_info}/\
data{data_version}-split={split_type}-ratio={train_ratio}/{sft_type}"
        metrics = load_metrics(file_dir, model_name, template_type)

        exist = False
        for item in metrics:
            if item["train_test_split"] == split_type and \
                item["train_ratio"] == train_ratio and item["lr"] == lr:

                update_item_metric(item)
                exist = True
                break
        if not exist:
            update_metrics(metrics, model_name, split_type, train_ratio, lr)
        metrics.sort(key=lambda x: (x["train_test_split"], x["train_ratio"], float(x["lr"])))
        save_metrics(file_dir, model_name, template_type, metrics, save=save)
    else:
        # 不同的ratio
        file_dir = f"test_metric_single_llm/{with_or_without_info}/\
data{data_version}-split={split_type}-sft={sft_type}-lr={lr}"
        metrics = load_metrics(file_dir, model_name, template_type)

        exist = False
        for item in metrics:
            if item["train_test_split"] == split_type and \
                item["train_ratio"] == train_ratio:

                update_item_metric(item)
                exist = True
                break
        if not exist:
            update_metrics(metrics, model_name, split_type, train_ratio)
        metrics.sort(key=lambda x: (x["train_test_split"], x["train_ratio"]))
        save_metrics(file_dir, model_name, template_type, metrics, save=save)
    return wrong_ans

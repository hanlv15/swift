import json

version_list = ["2024", "original"]


def load_train(version="2024"):
    """
    版本选择："2024" or "original"
    """
    
    if version == "2024":
        with open("/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train.json", "r") as f:
            return json.load(f)
    elif version == "original":
        with open("/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-main/data/train.json", "r") as f:
            return json.load(f)
    else:
        raise Exception(f"version需要从{version_list}中选择！")

def save_train(data, version="2024"):

    if version == "2024":
        with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()
    
def load_train_llm(version="2024", search_engine="brave"):

    if version == "2024" and search_engine == "brave":
        with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train_{search_engine}_search_llm.json", "r") as f:
            return json.load(f)
    else:
        raise Exception()

def save_train_llm(data, version="2024", search_engine="brave"):

    if version == "2024" and search_engine == "brave":
        with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train_{search_engine}_search_llm.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()

def load_train_search(version="2024", search_engine="brave"):

    if version == "2024" and search_engine == "brave":
        with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train_{search_engine}_search.json", "r") as f:
            return json.load(f)
    else:
        raise Exception()

def save_train_search(data, version="2024", search_engine="brave"):

    if version == "2024" and search_engine == "brave":
        with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-2024/train_{search_engine}_search.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        raise Exception()
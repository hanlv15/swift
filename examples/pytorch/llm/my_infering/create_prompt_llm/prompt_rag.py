import json
from tqdm import tqdm
from datetime import datetime


# import sys
# if "../../../../../../autogen" not in sys.path:
#     sys.path.append("../../../../../../autogen")
    
import autogen_label as al

def get_claim_with_date(claim, claim_date=None):
    if claim_date is None:
        return " " + claim
    
    res = "\n"
    res += "Publication date: " + claim_date + '\n' + "Content: " + claim
    return res

def get_bing_snippet(bing_search_results, K=5, claim_date=None):
    if bing_search_results.get("webPages") is None:
        return ""
    
    results = bing_search_results["webPages"]["value"]
    snippets = [i["snippet"] for i in results][:K]

    snippet = ""

    for i, item in enumerate(snippets):
        snippet += f"{i+1}. " + item + '\n'

    return snippet

def get_bing_snippet_v2(bing_search_results, K, claim_date, sort):
    """包含发布时间"""

    
    def get_snippets_dates(results, K=K):
        """找K个与claim发布时间最近的结果"""
        
        date_format = "%Y-%m-%d"
        date_claim = datetime.strptime(claim_date, date_format)
        
        if not sort:
            snippets = [i["snippet"] for i in results[:K]]
            dates = [i["datePublished"][:10] if i.get("datePublished") else "None" for i in results[:K]]
            return snippets, dates
        else:
            max_k = 20
            if K > max_k:
                raise Exception(f"K不能超过{max_k}")
            id_delta_list = [] # (id, delta): sinppet ID和snippet与claim发布日期的差
            results_10 = []
            need_supplement = False
            for i in results:
                if i.get("datePublished") is not None:
                    results_10.append(i)
                
                if len(results_10) == max_k:
                    break
            if len(results_10) < K:
                need_supplement = True

            for id, item in enumerate(results_10):
                date_item = datetime.strptime(item["datePublished"][:10], date_format)
                delta = abs((date_item - date_claim).days)

                id_delta_list.append((id, delta))

            id_delta_list_sorted = sorted(id_delta_list, key=lambda x: x[1])

            # print(id_delta_list_sorted)
            snippets = [results_10[i[0]]["snippet"] for i in id_delta_list_sorted[:K]]
            dates = [results_10[i[0]]["datePublished"][:10] for i in id_delta_list_sorted[:K]]

            
            if need_supplement:
                for i, item in enumerate(results):
                    if i in [x[0] for x in id_delta_list_sorted]:
                        continue
                    
                    snippets.append(item["snippet"])
                    dates.append("None")
                    if len(snippets) == K:
                        break

            return snippets, dates

    if bing_search_results.get("webPages") is None:
        return ""
    
    results = bing_search_results["webPages"]["value"]
    snippets, dates = get_snippets_dates(results)

    snippet = ""

    for i, item in enumerate(snippets):
        snippet += f"{i+1}.\n" + "Publication date: " + dates[i] + '\n' + "Content: " + item + '\n'

    return snippet

def get_brave_snippet(search_results, ids: slice, claim_date=None, sort=None):

    results = search_results["web"]["results"]
    snippet = ""
    cnt = 0
    for i, item in enumerate(results[ids]):

        date = item.get("page_age", "None")
        if date != "None":
            date = date[:10]

        extra_snippets = item.get("extra_snippets")
        if extra_snippets:
            content = "\n".join(extra_snippets)
        else:
            content = item["description"]
        snippet += f"Information {ids.start+i+1}:\n" + "Publication date: " + date + '\n' + \
            "Title: " + item["title"] + '\n' + "Content:\n" + content + '\n'
        cnt += 1
    return snippet, cnt

def get_prompt_for_generating_prior_knowledge(
        claim, claim_date, search_engine, search_results, 
        K=5, sort=False):
    """
    sort: 对search result 按时间进行排序
    """

    claim = claim.strip()
    pre = "Below is a CLAIM and some INFORMATION searched online. These pieces of INFORMATION are relevant to the CLAIM. This CLAIM and all INFORMATION include their respective publication dates and contents. To classify the CLAIM more accurately (if the content described by the CLAIM is correct, it will be classified as TRUE; if the content described by the CLAIM is incorrect, it will be classified as FALSE), please first expand on the given INFORMATION and provide a detailed summary of it. Then analyze, reason, and provide reasonable evidence to judge the correctness of the CLAIM based on the available information and your knowledge, and finally generate prior knowledge that helps classify the CLAIM.\n\n"

    text = "CLAIM:" + get_claim_with_date(claim, claim_date) +'\n\n'

    if search_engine == "bing":
        snippet = get_bing_snippet_v2(search_results, K=K, claim_date=claim_date, sort=sort)
    elif search_engine == "brave":
        snippet, _ = get_brave_snippet(search_results, ids=slice(0, K), claim_date=claim_date, sort=sort)
    else:
        raise Exception("Select search engines in [\"bing\", \"brave\"].")
    

    info = "INFORMATION:\n" + snippet

    return pre + text + info

def get_prompts_for_summarize_snippets(
        claim, claim_date, search_engine, search_results, K=5
):
    """
    以5条信息作为一个分组

    [0:5], [5:10], [5:15], [15:20]
    """
    slices_tmp = [slice(0, 5), slice(5, 10), slice(10, 15), slice(15, 20)]
    slices = []
    id_slices = None
    K = min(K, len(search_results["web"]["results"]))
    for i, _slice in enumerate(slices_tmp):
        if K-1 >= _slice.start and K - 1 < _slice.stop:
            id_slices = i
            break
    for i in range(id_slices + 1):
        if i < id_slices:
            slices.append(slices_tmp[i])
        else:
            slices.append(slice(slices_tmp[i].start, K))
    prompts = []
    claim = claim.strip()
    # pre = "Below is a CLAIM and some INFORMATION searched online. These pieces of INFORMATION are relevant to the CLAIM. This CLAIM and all INFORMATION include their respective publication dates and contents. To classify the CLAIM more accurately (if the content described by the CLAIM is correct, it will be classified as TRUE; if the content described by the CLAIM is incorrect, it will be classified as FALSE), please provide a detailed summary of each piece of INFORMATION given.\n\n"
    pre = "Please provide a detailed summary of each piece of information below"
    # pre = "Please summarize the information"
    # text = "CLAIM:" + get_claim_with_date(claim, claim_date) +'\n\n'

    for _slice in slices:
        if search_engine == "bing":
            snippet = get_bing_snippet_v2(search_results, K=K, claim_date=claim_date)
        elif search_engine == "brave":
            snippet, cnt = get_brave_snippet(search_results, ids=_slice, claim_date=claim_date)
        else:
            raise Exception("Select search engines in [\"bing\", \"brave\"].")
    
        info = "INFORMATION:\n" + snippet

        # prompts.append(pre + text + info)
        prompts.append(pre + f"(from No.{_slice.start + 1} to No.{_slice.start + cnt}).\n\n" + snippet)
    return prompts

def get_prompt_for_generating_prior_knowledge_by_summary(claim, claim_date, summary: list, K):
    
    if K == 5:
        _n = 1
    elif K == 10:
        _n = 2
    elif K == 15:
        _n = 3
    elif K == 20:
        _n = 4
    else:
        raise Exception()
    
    claim = claim.strip()
    pre = "Below is a claim and some information related to it. To classify the claim more accurately (if the content described by the claim is correct, it will be classified as TRUE; if the content described by the claim is incorrect, it will be classified as FALSE), please analyze, reason, and provide reasonable evidence to judge the correctness of the claim based on the available information and your knowledge, and finally generate prior knowledge that helps classify the claim.\n\n"

    text = "Claim:" + get_claim_with_date(claim, claim_date) +'\n\n'

    info = "Information:\n" + "\n\n".join(summary[:_n]).strip()

    return pre + text + info


def get_prompt_with_prior_knowledge(
        claim, search_engine, search_results, 
        prior_knowledge, K=5, claim_date=None, sort=False, known_info=True):
    """
    task + claim(with date) + prior knowledge with information

    sort: 对search result 按时间进行排序
    
    known_info: prior knowledge中是否包含已知信息
    """
    claim = claim.strip()
    pre = "According to the CLAIM and the PRIOR KNOWLEDGE, please classify the CLAIM as TRUE or FALSE. If the content described by the CLAIM is correct, then classify it as TRUE; if the content described by the CLAIM is incorrect, then classify it as FALSE.\n\n"
    text = "CLAIM:" + get_claim_with_date(claim, claim_date) +'\n\n'

    if search_engine == "bing":
        snippet = get_bing_snippet_v2(search_results, K=K, claim_date=claim_date, sort=sort)
    elif search_engine == "brave":
        snippet, _ = get_brave_snippet(search_results, ids=slice(0, K), claim_date=claim_date, sort=sort)
    else:
        raise Exception("Select search engines in [\"bing\", \"brave\"].")

    if known_info:
        return pre + text + "PRIOR KNOWLEDGE:\n" + snippet + '\n' + prior_knowledge.strip()
    else:
        return pre + text + "PRIOR KNOWLEDGE:\n" + prior_knowledge.strip()

def get_claim_id(claim, data_search):
    """
    非原始数据中的claim id，获取的是claim的位次编号
    """
    for i in range(len(data_search)):
        if claim.strip() in data_search[i]["claim"].strip():
            return i
        
def save_search_llm(x, search_engine):
    with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-main/data/train_{search_engine}_search_llm.json", "w") as f:
        json.dump(x, f, indent=4)

def save_search_summary(x, search_engine):
    with open(f"/home/hanlv/workspace/data/machine_learning/dataset/research/misinformation_dataset/COVMIS-main/data/train_{search_engine}_search_summary.json", "w") as f:
        json.dump(x, f, indent=4)

def save_search_llm_part(x, part):
    with open(f"train_search_llm_v{part}.json", "w") as f:
        json.dump(x, f, indent=4)

def load_search_llm_part(part):
    try:
        with open(f"train_search_llm_v{part}.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_search_summary_part(x, part):
    with open(f"train_search_summary_v{part}.json", "w") as f:
        json.dump(x, f, indent=4)

def load_search_summary_part(part):
    try:
        with open(f"train_search_summary_v{part}.json", "r") as f:
            return json.load(f)
    except:
        return []
    
def update_train_search_llm(
        model, model_name, port, search_engine, data_search, part, 
        K=5, sort=False):
    data_search_llm = load_search_llm_part(part)
    for i in tqdm(range(len(data_search_llm), len(data_search))):
        item = data_search[i]
        item_llm = {}
        
        item_llm["claim"] = item["claim"]
        prompt = get_prompt_for_generating_prior_knowledge(
            item["claim"], item["date"], search_engine, item[f"{search_engine}_search_results"], K=K, sort=sort)
        
        item_llm[f"prior_knowledge_{model_name}"] = al.get_response(prompt, model, cache_seed=None, port=port)
        data_search_llm.append(item_llm)

        if i % 20 == 0:
            save_search_llm_part(data_search_llm, part)

    save_search_llm_part(data_search_llm, part)

def update_train_search_summary(
        model, model_name, port, search_engine, data_search, part, K=20):
    data_search_summary = load_search_summary_part(part)
    for i in tqdm(range(len(data_search_summary), len(data_search))):
        item = data_search[i]
        item_summary = {}
        
        # print(item["claim"])
        item_summary["claim"] = item["claim"]
        res = []
        prompts = get_prompts_for_summarize_snippets(
            item["claim"], item["date"], search_engine, item[f"{search_engine}_search_results"], K=K)
        for prompt in prompts:
            res_tmp = al.get_response(prompt, model, cache_seed=None, port=port)
            res.append(res_tmp.strip())
        
        item_summary[f"summary_{model_name}"] = res
        
        data_search_summary.append(item_summary)

        if i % 5 == 0:
            save_search_summary_part(data_search_summary, part)

    save_search_summary_part(data_search_summary, part)

def update_train_search_llm_by_summary(
        model, model_name, port, 
        data_search_summary, summary_version, part, K=5):
    
    assert K in [5, 10, 15, 20], "请从K只能为5, 10, 15, 20。"

    data_search_llm = load_search_llm_part(part)
    for i in tqdm(range(len(data_search_llm), len(data_search_summary))):
        item = data_search_summary[i]
        item_llm = {}
        
        item_llm["claim"] = item["claim"]
        prompt = get_prompt_for_generating_prior_knowledge_by_summary(
            item["claim"], item["date"], item[f"summary_solar_v{summary_version}"], K=K)
        
        item_llm[f"prior_knowledge_by_summary_{model_name}"] = al.get_response(prompt, model, cache_seed=None, port=port)
        data_search_llm.append(item_llm)

        if i % 20 == 0:
            save_search_llm_part(data_search_llm, part)

    save_search_llm_part(data_search_llm, part)


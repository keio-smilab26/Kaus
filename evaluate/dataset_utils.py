import os
import json
import numpy as np
from datasets import load_dataset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.abspath(os.path.join(_REPO_ROOT, "..", "datasets"))

PATH_MAP = {
    "flickr8k_root": os.path.join(DATA_ROOT, "flickr8k"),
    "flickr30k_root": os.path.join(DATA_ROOT, "flickr30k"),
    "coco_root": os.path.join(DATA_ROOT, "coco"),
    "polaris_root": os.path.join(DATA_ROOT, "polaris/polaris"),
    "polaris_exp_json": os.path.join(_REPO_ROOT, "ascella_dataset", "polaris.json"),
    "nebula_exp_json":  os.path.join(_REPO_ROOT, "ascella_dataset", "nebula.json"),
}

def get_expert_dataset(dataset_alias, args=None):
    records = []
    
    # 1. Nebula (Hugging Faceから取得)
    if dataset_alias == "nebula":
        raw_ds = load_dataset("Ka2ukiMatsuda/Nebula", split="test", streaming=False)
        for d in raw_ds:
            records.append({
                "image": d['image'], "mt": d['mt'], "gold": float(d['human_score']),
                "id": d.get('file_name') or d.get('imgid')
            })

    # 2. Polaris
    elif dataset_alias == "polaris":
        json_path = os.path.join(PATH_MAP["polaris_root"], "polaris_test.csv")
        raw_ds = load_dataset("csv", data_files=json_path, split="train")
        for row in raw_ds:
            img_path = os.path.join(PATH_MAP["polaris_root"], "images", row['imgid'])
            records.append({
                "image": img_path, "mt": row['mt'], "gold": float(row['score']), "id": row['imgid']
            })

    # 3. Flickr8k 関連 (エイリアスによってパスを切り替え)
    elif dataset_alias in ["flickr8k-ex", "flickr8k-cf"]:
        if dataset_alias == "flickr8k-ex":
            json_path = os.path.join(PATH_MAP["flickr8k_root"], "flickr8k.json")
        else:
            json_path = os.path.join(PATH_MAP["flickr8k_root"], "crowdflower_flickr8k.json")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_dir = os.path.join(PATH_MAP["flickr8k_root"], "Images")
        for img_id, content in data.items():
            img_filename = os.path.basename(content['human_judgement'][0]['image_path'])
            img_path = os.path.join(img_dir, img_filename)
            for hj in content['human_judgement']:
                if np.isnan(hj.get('rating', np.nan)): continue
                if dataset_alias == "flickr8k-cf":
                    # cf ratings are already in [0, 1] (averaged binary annotations)
                    score = float(hj['rating'])
                else:
                    # ex ratings are integers 1–4 → normalize to [0, 1]
                    score = (float(hj['rating']) - 1.0) / 3.0
                records.append({
                    "image": img_path, "mt": hj['caption'], "gold": score, "id": img_id
                })

    # 4. Composite
    elif dataset_alias == "composite":
        json_path = os.path.join(DATA_ROOT, "composite", "composite.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dir_map = {
            "flickr8k": os.path.join(PATH_MAP["flickr8k_root"], "Images"),
            "flickr30k": os.path.join(PATH_MAP["flickr30k_root"], "images"),
            "coco": os.path.join(PATH_MAP["coco_root"], "val2014")
        }
        for ds_key, items in data.items():
            base_dir = dir_map.get(ds_key)
            for item in items:
                img_path = os.path.join(base_dir, os.path.basename(item['image']))
                score = (float(item['human']) - 1.0) / 4.0
                records.append({
                    "image": img_path, "mt": item['caption'], "gold": score,
                    "id": item['image']  # e.g. "Flickr8k_Dataset/909191414_1cf5d85821.jpg"
                })

    # 5. Polaris-exp train
    elif dataset_alias == "polaris-exp-train":
        with open(PATH_MAP["polaris_exp_json"], 'r', encoding='utf-8') as f:
            entries = json.load(f)
        img_dir = os.path.join(PATH_MAP["polaris_root"], "images")
        for e in entries:
            img_path = os.path.join(img_dir, e["image_id"])
            records.append({
                "image": img_path, "mt": e["caption"], "gold": float(e["score"]),
                "id": e["image_id"], "reasoning": e.get("qwen_reasoning", ""),
            })

    # 6. Nebula-exp train
    elif dataset_alias == "nebula-exp-train":
        with open(PATH_MAP["nebula_exp_json"], 'r', encoding='utf-8') as f:
            entries = json.load(f)
        print("Loading Nebula HF dataset for image lookup...")
        hf_ds = load_dataset("Ka2ukiMatsuda/Nebula", split="train", streaming=False)
        imgid_to_image = {item["file_name"]: item["image"] for item in hf_ds}
        n_missing = 0
        for e in entries:
            if e["image_id"] not in imgid_to_image:
                n_missing += 1
                continue
            records.append({
                "image": imgid_to_image[e["image_id"]], "mt": e["caption"],
                "gold": float(e["score"]), "id": e["image_id"],
                "reasoning": e.get("qwen_reasoning", ""),
            })
        if n_missing:
            print(f"[nebula-exp-train] skipped {n_missing} entries with no HF image")

    # 互換性維持
    for r in records:
        r["caption"] = r.get("mt")
    return records
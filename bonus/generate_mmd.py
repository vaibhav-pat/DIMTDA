import os
import json

def create_mmd_files(json_path, en_out_dir, zh_out_dir):
    os.makedirs(en_out_dir, exist_ok=True)
    os.makedirs(zh_out_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        image_name = entry["image"].replace(".png", "")
        en_text = entry.get("src_text", "").strip()
        zh_text = entry.get("tgt_text", "").strip()

        # Save .mmd files
        with open(os.path.join(en_out_dir, f"{image_name}.mmd"), "w", encoding="utf-8") as enf:
            enf.write(en_text)
        with open(os.path.join(zh_out_dir, f"{image_name}.mmd"), "w", encoding="utf-8") as zhf:
            zhf.write(zh_text)

    print(f"✅ Finished: {len(data)} files written to {en_out_dir} and {zh_out_dir}")

# Run this for train.json
create_mmd_files(
    "DIMT_dataset/train.json",
    "data/en_mmd",
    "data/zh_mmd"
)

# Run again for valid.json (same format)
create_mmd_files(
    "DIMT_dataset/valid.json",
    "data/en_mmd",
    "data/zh_mmd"
)

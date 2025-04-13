import json

def generate_split_json(train_json_path, valid_json_path, output_path):
    def get_names(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [entry["image"].replace(".png", "") for entry in data]

    train_list = get_names(train_json_path)
    valid_list = get_names(valid_json_path)

    split_dict = {
        "train_name_list": train_list,
        "valid_name_list": valid_list
    }

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(split_dict, out, indent=2, ensure_ascii=False)

    print(f"✅ split.json created with {len(train_list)} train and {len(valid_list)} valid entries.")

# Run it
generate_split_json(
    "DIMT_dataset/train.json",
    "DIMT_dataset/valid.json",
    "data/split.json"
)

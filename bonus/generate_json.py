import os
import json
from PIL import Image
import pytesseract
from googletrans import Translator

translator = Translator()

def process_folder(img_dir, output_json_path):
    results = []
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])

    for i, filename in enumerate(files):
        img_path = os.path.join(img_dir, filename)
        try:
            image = Image.open(img_path)
            src_text = pytesseract.image_to_string(image).strip()

            tgt_text = translator.translate(src_text, src='en', dest='zh-cn').text.strip() if src_text else ""
        except Exception as e:
            print(f"⚠️ Error with {filename}: {e}")
            src_text, tgt_text = "", 
        
        results.append({
            "image": filename,
            "src_text": src_text,
            "tgt_text": tgt_text
        })

        if i % 100 == 0:
            print(f"🌀 Processed {i}/{len(files)}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done: {output_json_path} written with {len(results)} entries.")

# Change these paths to your actual dataset
process_folder("DIMT_dataset/trainset", "DIMT_dataset/train.json")
process_folder("DIMT_dataset/validset", "DIMT_dataset/valid.json")
process_folder("DIMT_dataset/testset", "DIMT_dataset/test.json")
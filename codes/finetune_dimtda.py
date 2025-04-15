import os
import json
import torch
import jieba
import re
import argparse

def train(args):
    MAX_LENGTH = args.max_length

    from transformers import AutoTokenizer, DonutProcessor, BeitImageProcessor
    dit_processor = BeitImageProcessor.from_pretrained(args.dit_model_dir)
    nougat_processor = DonutProcessor.from_pretrained(args.image_processor_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    from my_dataset import DoTADataset
    valid_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADataset(dit_processor, nougat_processor, zh_tokenizer, args.image_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)

    from transformers import EncoderDecoderModel, VisionEncoderDecoderModel, BeitModel, EncoderDecoderConfig
    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    dit_model = BeitModel.from_pretrained(args.dit_model_dir)
    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir)

    from my_model import DIMTDAModel
    my_config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = DIMTDAModel(my_config, trans_model, dit_model, nougat_model, args.num_queries, args.qformer_config_dir)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Dataloaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.dataloader_num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.dataloader_num_workers)

    # Optimizer & Scheduler
    from transformers import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{args.num_train_epochs}] Training Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valid_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100

                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print("Saved best model.")

    # Save final accuracy
    with open(os.path.join(args.output_dir, "accuracy.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--dit_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--image_processor_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--qformer_config_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=1024)
    
    args = parser.parse_args()
    
    train(args)

import os
import json
import torch
import jieba
import re
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EncoderDecoderModel, EncoderDecoderConfig, BertConfig
from torch.nn import CrossEntropyLoss
from my_dataset import DoTADatasetTrans
from tqdm import tqdm

def calculate_accuracy(preds, labels, pad_token_id):
    preds = preds.argmax(dim=-1)
    correct = (preds == labels) & (labels != pad_token_id)
    total = (labels != pad_token_id)
    accuracy = correct.sum().item() / total.sum().item()
    return accuracy

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = args.max_length

    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']

    train_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)
    valid_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.dataloader_num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.dataloader_num_workers)

    encoder_config = BertConfig()
    decoder_config = BertConfig(is_decoder=True, add_cross_attention=True)
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    encoder_decoder_config.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.max_length = MAX_LENGTH
    encoder_decoder_config.vocab_size = len(zh_tokenizer)

    model = EncoderDecoderModel(config=encoder_decoder_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=zh_tokenizer.pad_token_id)

    step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss, total_acc = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            acc = calculate_accuracy(logits, labels, zh_tokenizer.pad_token_id)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_acc += acc
            step += 1

            loop.set_postfix(loss=loss.item(), acc=acc)

            if step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{step}.pth")
                torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch+1} - Avg Loss: {total_loss/len(train_loader):.4f} | Avg Acc: {total_acc/len(train_loader):.4f}")

    final_model_path = os.path.join(args.output_dir, "model_pretrained.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved at {final_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_tokenizer_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--en_mmd_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

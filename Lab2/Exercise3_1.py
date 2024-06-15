import torch.nn as nn
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from datasets import load_dataset
from tqdm import tqdm
import wandb
import argparse


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, text, device):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(device)
        output = self.model(**inputs)
        cls_token = output['last_hidden_state'][:, 0, :]
        return self.head(cls_token)


def train(model, train_dataset, epochs=10, lr=3e-4, fine_tune=False, device="cpu", use_wandb=False):
    model.train()
    if fine_tune:
        params = [{"params": model.model.parameters(), "lr": lr / 100},
                  {"params": model.head.parameters(), "lr": lr}
                  ]
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        acc_loss = 0
        for i, batch in tqdm(enumerate(train_dataset)):
            text = batch["text"]
            label = batch["label"]
            label = label.to(device)
            optimizer.zero_grad()
            output = model(text, device)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # accumulate the loss
            acc_loss += loss.item()
            if (i + 1) % 500 == 0:
                if use_wandb:
                    wandb.log({"loss": acc_loss / 500})
                print(f"Epoch {epoch}, batch {i}, loss {acc_loss / 500}")
                acc_loss = 0

def test(model, dataset, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset)):
            text = batch["text"]
            label = batch["label"]
            label = label.to(device)
            output = model(text, device)
            predicted = torch.argmax(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct, total


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--training", choices=['zero_shot', 'train_head', 'finetune'])
    argparser.add_argument("--wandb", action="store_true")
    args = argparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tweet_eval = load_dataset("tweet_eval", 'emotion')
    tweet_train = tweet_eval["train"]
    num_classes = len(set(tweet_train['label']))
    tweet_test = tweet_eval["test"]

    dl_train = torch.utils.data.DataLoader(tweet_train, batch_size=8, shuffle=True)
    dl_test = torch.utils.data.DataLoader(tweet_test, batch_size=8, shuffle=False)
    model = BERTClassifier(num_classes).to(device)
    epochs = 1

    # zero shot accuracy
    if args.training == 'zero_shot':
        correct, total = test(model, dl_test, device=device)
        print(f"Zero shot accuracy: {correct / total}")
        if args.wandb:
            wandb.init(project="Text classification", name=args.training, config={"strategy": args.training})
            wandb.log({"accuracy": correct / total})
            wandb.finish()

    else:
        if args.training == 'train_head':
            train(model, dl_train, epochs=1, device=device, fine_tune=False, use_wandb=args.wandb)
        else:
            train(model, dl_train, epochs=1, device=device, fine_tune=True, use_wandb=args.wandb)
        correct, total = test(model, dl_test, device=device)
        print(f"Zero shot accuracy: {correct / total}")
        if args.wandb:
            wandb.init(project="Text classification", name=args.training, config={"strategy": args.training})
            wandb.log({"accuracy": correct / total})
            wandb.finish()


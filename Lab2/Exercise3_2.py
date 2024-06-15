from datasets import load_dataset, load_metric
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np

def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                        enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
def parse_args():
    parser = argparse.ArgumentParser(description="Question answering model")
    parser.add_argument("--train", action="store_true", help=" If set, it trains the model on the SWAG dataset")
    parser.add_argument("--model-path", type=str, help="Path to the model to use for prediction")
    parser.add_argument("--question", type=str, help="Write here the question you are want to ask to the model")
    parser.add_argument("-n", "--answers", nargs="+", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_checkpoint = 'bert-base-uncased'
    batch_size = 4
    if args.train:
        datasets = load_dataset("swag", "regular")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        ending_names = ["ending0", "ending1", "ending2", "ending3"]

        encoded_datasets = datasets.map(preprocess_function, batched=True)
        model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)


        def compute_metrics(eval_predictions):
            predictions, label_ids = eval_predictions
            preds = np.argmax(predictions, axis=1)
            return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

        model_name = model_checkpoint.split("/")[-1]
        args = TrainingArguments(
            f"./results/exercise3_2/{model_name}-finetuned-swag",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=False,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )
        trainer.train()

    if args.question and args.answers and args.model_path:
        n_answers = len(args.answers)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        question_answer = [[args.question, answer] for answer in args.answers]
        inputs = tokenizer(question_answer, return_tensors="pt", padding=True)
        labels = torch.tensor(0).unsqueeze(0)

        model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        print(f"Model answer: {args.answers[predictions]}")

        with open("./results/exercise3_2/interactions.txt", "a") as f:
            f.write(f"{args.question}\n")
            for i, answer in enumerate(args.answers):
                f.write(f"{i}: {answer}\n")
            f.write(f"Model answer: {args.answers[predictions]}\n\n\n")







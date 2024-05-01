"""
This module is an implmentation of ViT for Image Classification and is derived
from a tutorial on Hugging Face.
https://huggingface.co/blog/fine-tune-vit
"""

import os
import tqdm
import torch
import argparse
import numpy as np
from datetime import datetime
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTForImageClassification


def transform(example_batch):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


def process_example(example):
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def make_file(true_labels, pred_labels) -> None:
    """
    This function makes a txt file of the filename, true labels (gathered from the filename) and the predicted labels.
    :param true_labels: List of true labels.
    :param pred_labels: List of predicted labels.
    :return: None
    """
    outdir = "./output_label_files"
    save_file_name = os.path.join(outdir, f"vit_classification_labels_{datetime.now()}.txt")

    with open(save_file_name, "w") as f:
        f.write("true_label,predicted_label\n")

        for gt, p in zip(true_labels, pred_labels):
            f.write(f"{gt},{p}\n")


if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, type=str, help="The input directory path.")
    ap.add_argument("-o", "--output_dir", required=True, type=str, help="The output directory path.")
    ap.add_argument("-m", "--mode", required=True, type=str,
                    help="Determine which mode to run the model: 'train' or 'test'")
    ap.add_argument("-w", "--weights", required=False, default="./vit_runs/new_vit/checkpoint-9300/", type=str,
                    help="The path to the weights.")
    opts = vars(ap.parse_args())

    # Required Variables
    mode = opts['mode']
    data_dir = opts['input_dir']
    output_dir = opts['output_dir']
    weights_path = opts['weights']
    model_path = 'google/vit-base-patch16-224-in21k'

    # Check if input path is valid
    if not os.path.exists(data_dir):
        print(f"ERROR: '{data_dir}' does not exist!")
        exit()

    # Check if weights path is valid
    if mode == 'test':
        if not os.path.exists(weights_path):
            print(f"ERROR: Can't find weights at following path: {weights_path}")

    # Select a device
    print(f"--- GPU Available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if mode == "train":
        # Getting dataset and labels
        dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)
        labels = dataset['train'].features['label'].names
        prep_dataset = dataset.with_transform(transform)

        id2label = {str(i): c for i, c in enumerate(labels)}
        label2id = {c: str(i) for i, c in enumerate(labels)}

        # Model definition
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )

        # Training Configurations
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=4,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prep_dataset["train"],
            eval_dataset=prep_dataset["validation"],
            tokenizer=processor
        )

        # Train Model
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        # Evaluate the Model
        metrics = trainer.evaluate(prep_dataset['validation'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        print("Done.")

    else:
        # Get dataset
        dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)

        # Required Variables
        true_labels = []
        predicted_labels = []

        # Model definition
        processor = ViTImageProcessor.from_pretrained(weights_path)
        model = ViTForImageClassification.from_pretrained(weights_path)

        # Predict Class Labels
        for d in tqdm.tqdm(dataset['train']):

            inputs = processor(d['image'], return_tensors='pt')
            true_labels.append(model.config.id2label[d['label']])

            with torch.no_grad():
                logits = model(**inputs).logits

                pred_label = logits.argmax(-1).item()
                predicted_labels.append(model.config.id2label[pred_label])

        make_file(true_labels, predicted_labels)
        print("Done.")

"""
This module is an implementation of Google's Efficientnet derived from a
tutorial on Hugging Face.
https://huggingface.co/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification
"""

import os
import tqdm
import torch
import argparse
from datasets import load_dataset
from torchvision import transforms
from transformers import TrainingArguments, Trainer, AutoTokenizer
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification, AutoImageProcessor


def preprocess_fn(example):
    tokenizer = EfficientNetImageProcessor("google/efficientnet-b7")
    inputs = tokenizer(example['image'], return_tensors='pt')
    inputs['label'] = example['label']
    return inputs


def transform(example_batch):
    tokenizer = EfficientNetImageProcessor("google/efficientnet-b7")
    inputs = tokenizer([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def make_file(files, true_labels, pred_labels) -> None:
    """
    This function makes a txt file of the filename, true labels (gathered from the filename) and the predicted labels.
    :param files: List of filenames.
    :param true_labels: List of true labels.
    :param pred_labels: List of predicted labels.
    :return: None
    """
    outdir = "./output_label_files"
    save_file_name = os.path.join(outdir, f"yolo8_classification_labels_{datetime.now()}.txt")

    with open(save_file_name, "w") as f:
        f.write("file_name,true_label,predicted_label\n")

        for fn, gt, p in zip(files, true_labels, pred_labels):
            f.write(f"{fn},{gt},{p}\n")


if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, type=str, help="The input directory path.")
    ap.add_argument("-o", "--output_dir", required=True, type=str, help="The output directory path.")
    ap.add_argument("-m", "--mode", required=True, type=str,
                    help="Determine which mode to run the model: 'train' or 'test'")
    ap.add_argument("-w", "--weights", required=False, default=None, type=str, help="The path to the weights.")
    opts = vars(ap.parse_args())

    # Required Variables
    mode = opts['mode']
    data_dir = opts['input_dir']
    output_dir = opts['output_dir']
    weights_path = opts['weights']
    model_path = "google/efficientnet-b7"

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

    # Getting dataset and labels
    dataset = load_dataset("imagefolder", data_dir=data_dir, drop_labels=False)
    labels = dataset['train'].features['label'].names

    tokenizer = EfficientNetImageProcessor(model_path)

    # Setting up class label conversion
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}

    if mode == "train":
        # TRAIN MODEL =========================================================
        # Model definition
        # feature_extractor = EfficientNetFeatureExtractor.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = EfficientNetForImageClassification.from_pretrained(model_path, num_labels=len(labels),
                                                                   id2label=id2label, label2id=label2id,
                                                                   ignore_mismatched_sizes=True)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=100,
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="tensorboard"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=collate_fn
        )

        # Train the model
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)
        print("Done.")

    else:
        # EVALUATE MODEL ======================================================
        # Get dataset
        dataset = load_dataset("imagefolder", data_dir=data_dir)

        # Required variables
        true_labels = []
        predicted_labels = []

        # Model Definition
        processor = AutoImageProcessor.from_pretrained(weights_path)
        model = EfficientNetForImageClassification.from_pretrained(weights_path)

        # Predict Class label
        for d in tqdm.tqdm(dataset):
            inputs = processor(d, return_tensors='pt')

            with torch.no_grad():
                logits = model(**inputs).logits

            pred_label = logits.argmax(-1).item()
            predicted_labels.append(model.config.id2label[pred_label])

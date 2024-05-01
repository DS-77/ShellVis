import os
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification

if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, type=str, help="The path to the dataset directory.")
    opts = vars(ap.parse_args())

    data_path = opts['dataset']
    output_dir = "./vit_runs/"

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Check if data path is valid.
    if not os.path.exists(data_path):
        print(f"ERROR: '{data_path}' doesn't exist!")
        exit()

    # Get the dataset
    train, val, id2label, label2id = VisionDataset.fromImageFolder(data_path, test_ratio=0.10, balanced=True,
                                                                   augmentation=True)

    # Model definition
    huggingface_model = 'google/vit-base-patch16-224-in21k'
    model = ViTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(huggingface_model)

    # Set up the trainer
    trainer = VisionClassifierTrainer(
        model_name="ViTShellVis",
        train=train,
        test=val,
        output_dir=output_dir,
        max_epochs=100,
        batch_size=16,
        lr=2e-5,
        fp16=True,
        model=model,
        feature_extractor=feature_extractor
    )

    # Train the model
    ref, hyp = trainer.evaluate_f1_score()

    # Confusion Matrix
    # cm = confusion_matrix(ref, hyp)
    # labels = list(label2id.keys())
    # df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    #
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
    # plt.savefig(f"{output_dir}conf_matrix_1.png")
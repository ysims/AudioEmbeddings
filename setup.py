import argparse
import torch


def setup():
    parser = argparse.ArgumentParser()

    # Arguments for the model
    parser.add_argument(
        "--dataset", type=str, default="ESC-50", help="Dataset to train on."
    )
    parser.add_argument(
        "--split", type=str, default="fold04", help="Class split for the chosen dataset."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on. Auto will check if cuda can be used, else it will use cpu.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=3000,
        help="Number of epochs to run, or max if early stopping is on.",
    )
    parser.add_argument(
        "--meta_epochs",
        type=int,
        default=20,
        help="Number of epochs to run, or max if early stopping is on.",
    )
    parser.add_argument(
        "--meta_contrast_epoch",
        type=int,
        default=20,
        help="Number of epochs to run, or max if early stopping is on.",
    )
    parser.add_argument(
        "--meta_classify_epoch",
        type=int,
        default=20,
        help="Number of epochs to run, or max if early stopping is on.",
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        default=True,
        help="Whether to stop early when validation loss isn't improving.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="How many epochs to watch for early stopping.",
    )
    parser.add_argument(
        "--patience_meta",
        type=int,
        default=10,
        help="How many epochs to watch for early stopping in the meta-learning loop.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory for the tensorboard logs.",
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default="",
        help="Directory of a pretrained model.",
    )
    parser.add_argument(
        "--emb_size", type=int, default=128, help="Size of the audio embeddings."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="Batch size for training."
    )
    parser.add_argument(
        "--supcon_weight", type=int, default=0.2, help="Weight of supervised contrastive loss during combined training."
    )
    parser.add_argument(
        "--con_batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="YAMNet",
        choices=["VGGish", "YAMNet", "TYAMNet", "YAMNet3", "Inception"],
        help="Model architecture to training.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # If ESC-50, then set the path and classes
    if args.dataset == "ESC-50":
        args.data_path = "../data/ESC-50/audio"

        # fmt: off
        # Define the classes based on the splits; the split in the var name is the one removed for 5-fold cross-validation
        if args.split == "cat0":
            args.train_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        elif args.split == "cat1":
            args.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        elif args.split == "cat2":
            args.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        elif args.split == "cat3":
            args.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        elif args.split == "cat4":
            args.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        elif args.split == "fold0":
            args.train_classes = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34, 36, 37, 39, 41, 42, 43, 44, 45, 47, 49]
        elif args.split == "fold1":
            args.train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 40, 41, 43, 44, 45, 46, 47, 48]
        elif args.split == "fold2":
            args.train_classes = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47, 48, 49]
        elif args.split == "fold3":
            args.train_classes = [0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 48, 49]
        elif args.split == "fold4" or args.split == "test":
            args.train_classes = [1, 2, 3, 4, 6, 7, 10, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49]       
        # These are for testing with fold 4, and doing 4-fold cross-validation on the remaining folds
        elif args.split == "fold04":
            args.train_classes = [1, 4, 6, 7, 10, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34, 36, 39, 41, 42, 44, 45, 47, 49]
        elif args.split == "fold14": 
            args.train_classes = [1, 2, 3, 4, 6, 7, 10, 14, 17, 18, 20, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 38, 40, 41, 44, 45, 46, 47, 48]
        elif args.split == "fold24":
            args.train_classes = [1, 2, 3, 6, 7, 13, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 38, 39, 40, 42, 44, 46, 47, 48, 49]
        elif args.split == "fold34":
            args.train_classes = [2, 3, 4, 10, 13, 14, 17, 19, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 45, 46, 48, 49]
        # fmt: on

    # If FSC22, then set the path and classes
    elif args.dataset == "FSC22":
        args.data_path = "../data/FSC22/audio"

        # fmt: off
        args.train_classes = [0, 1, 2, 3, 4, 10, 11, 14, 16, 19, 20, 24, 25]
        # args.val_classes = [6, 8, 9, 12, 13, 18, 22]
        # args.test_classes = [5, 7, 15, 17, 21, 23, 26]
        # fmt: on

    args.save_path = f"./checkpoints/{args.model}_{args.dataset}_{args.split}.pt"

    return args

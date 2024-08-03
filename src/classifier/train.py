import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


class SqueezeNetBranch(nn.Module):
    def __init__(self, pretrained=False):
        super(SqueezeNetBranch, self).__init__()
        self.model = models.squeezenet1_0(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)


class LSTMBranch(nn.Module):
    def __init__(
        self, hidden_size=256, num_layers=2, pretrained=False, pretrained_path=None
    ):
        super(LSTMBranch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.lstm = None
        self.classifier = None

    def forward(self, x):
        if self.lstm is None:
            input_size = x.size(2)
            self.lstm = nn.LSTM(
                input_size, self.hidden_size, self.num_layers, batch_first=True
            )
            self.classifier = nn.Linear(self.hidden_size, 3)
            if self.pretrained and self.pretrained_path:
                self._load_pretrained_weights(self.pretrained_path)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.classifier(out[:, -1, :])

    def _load_pretrained_weights(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class SiameseNetwork(nn.Module):
    def __init__(
        self,
        label_structure,
        pretrained_squeezenet=False,
        cnn_pretrained_path=None,
        pretrained_lstm=False,
        lstm_pretrained_path=None,
    ):
        super(SiameseNetwork, self).__init__()
        if pretrained_squeezenet == False or cnn_pretrained_path == "default":
            self.squeeze_branch = SqueezeNetBranch(pretrained=pretrained_squeezenet)
        else:
            self.squeeze_branch = SqueezeNetBranch(pretrained=False)
            self.squeeze_branch.load_state_dict(torch.load(cnn_pretrained_path))

        self.lstm_branch = LSTMBranch(
            pretrained=pretrained_lstm, pretrained_path=lstm_pretrained_path
        )

        # Placeholder for input size of the classifiers, which will be set dynamically
        self.combined_feat_size = None
        self.label_classifier = None
        self.sublabel_classifiers = nn.ModuleDict()

        # Store the number of sublabels for each label
        self.num_sublabels = {
            label: len(sublabels) for label, sublabels in label_structure.items()
        }

    def forward(self, img, seq):
        img_feat = self.squeeze_branch(img)
        seq_feat = self.lstm_branch(seq)

        combined_feat = torch.cat((img_feat, seq_feat), dim=1)

        # Dynamically set the input size for the classifier layers if not set
        if self.combined_feat_size is None:
            self.combined_feat_size = combined_feat.size(1)
            self.label_classifier = nn.Linear(
                self.combined_feat_size, len(self.num_sublabels)
            )
            for label in self.num_sublabels:
                self.sublabel_classifiers[label] = nn.Linear(
                    self.combined_feat_size, self.num_sublabels[label]
                )

        label_out = self.label_classifier(combined_feat)

        sublabel_outs = {}
        for label in self.sublabel_classifiers:
            sublabel_outs[label] = self.sublabel_classifiers[label](combined_feat)

        return label_out, sublabel_outs


class Preprocess:
    def __init__(self, config):
        self.config = config

    def _load_imgs(self, paths):
        """
        Load images from given paths and preprocess them.
        """
        images = []
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize to the required size
                img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                img = np.transpose(img, (2, 0, 1))  # Change to CxHxW format
                images.append(
                    torch.tensor(img, dtype=torch.float32)
                )  # Explicitly set type to float32
            else:
                images.append(
                    torch.zeros(
                        3, 224, 224, dtype=torch.float32
                    )  # Placeholder for missing images
                )
        return images

    def _encode_string(self, row):
        """
        Encode a mix of strings and numerical values.
        """
        encoded_values = []
        for col in row.index:
            if isinstance(row[col], str):  # If the column is of string type
                le = OneHotEncoder(sparse_output=False)
                encoded_col = le.fit_transform([[row[col]]])[0]
                encoded_values.extend(encoded_col)
            else:
                encoded_values.append(row[col])

        return torch.tensor(encoded_values, dtype=torch.float32)

    def run(self, df):
        # Load images
        img_paths = df["Name"].apply(
            lambda name: os.path.join(
                self.config.path.intermediate.graph, f"{name}.png"
            )
        )
        images = self._load_imgs(img_paths)

        # Encode sequences
        seq_data = df.drop(columns=["Name", "Label", "Sublabel"])
        encoded_seq_data = seq_data.apply(self._encode_string, axis=1).tolist()

        # Normalize sequences
        scaler = StandardScaler()
        sequences = scaler.fit_transform(encoded_seq_data).astype(
            np.float32
        )  # Ensure type is float32

        # Extract labels and sublabels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(
            np.array(df["Label"]).reshape(-1, 1)
        ).astype(np.float32)
        sublabel_encoder = LabelEncoder()
        sublabels = sublabel_encoder.fit_transform(
            np.array(df["Sublabel"]).reshape(-1, 1)
        ).astype(np.float32)

        # Convert labels and sublabels to tensors
        labels = torch.tensor(labels, dtype=torch.float32)
        sublabels = torch.tensor(sublabels, dtype=torch.float32)

        # Create dataset
        dataset = list(zip(images, sequences, labels, sublabels))

        # Split dataset into train, val, and test sets
        train_data, test_data = train_test_split(
            dataset,
            test_size=(1 - self.config.hyperparameter.train_ratio),
            random_state=42,
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=(1 - self.config.hyperparameter.validate_ratio),
            random_state=42,
        )

        return train_data, val_data, test_data


class Train:
    def __init__(self, config, label_structure, data, train_id):
        self.config = config
        self.label_structure = label_structure
        self.data = data
        self.train_id = train_id

    def _save_model(self, model, epoch):
        torch.save(
            model.squeeze_branch.state_dict(),
            os.path.join(
                self.config.path.train.model,
                f"{self.config.hyperparameter.model.CNN.model}_branch_epoch_{epoch}_{self.train_id}.pth",
            ),
        )
        torch.save(
            model.lstm_branch.state_dict(),
            os.path.join(
                self.config.path.train.model,
                f"lstm_branch_epoch_{epoch}_{self.train_id}.pth",
            ),
        )

    def _run_phase(self, phase, loader, model, criterion, optimizer, device):
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_labels = []
        all_preds = []

        for imgs, seqs, labels, sublabels in tqdm(loader, desc=phase.capitalize()):
            imgs, seqs, labels, sublabels = (
                imgs.to(device, dtype=torch.float32),
                seqs.to(device, dtype=torch.float32).unsqueeze(1),  # Ensure seqs is 3D
                labels.to(device, dtype=torch.long),  # Convert labels to LongTensor
                sublabels.to(
                    device, dtype=torch.long
                ),  # Convert sublabels to LongTensor
            )

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                label_out, sublabel_outs = model(imgs, seqs)

                loss_label = criterion(label_out, labels)
                loss_sublabel = 0
                for i in range(len(sublabels)):
                    label_index = labels[i].item()
                    if label_index in sublabel_outs:
                        sublabel_pred = sublabel_outs[label_index]
                        sublabel_true = sublabels[i]
                        loss_sublabel += criterion(sublabel_pred, sublabel_true)

                loss = loss_label + loss_sublabel

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(label_out, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=1
        )
        epoch_recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=1
        )
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)

        return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

    def run(self):
        preprocess = Preprocess(self.config)
        train_data, val_data, test_data = preprocess.run(self.data)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SiameseNetwork(
            self.label_structure,
            pretrained_squeezenet=self.config.hyperparameter.model.CNN.pretrained,
            cnn_pretrained_path=self.config.hyperparameter.model.CNN.pretrained_model_path,
            pretrained_lstm=self.config.hyperparameter.model.LSTM.pretrained,
            lstm_pretrained_path=self.config.hyperparameter.model.LSTM.pretrained_model_path,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.config.hyperparameter.learning_rate
        )
        with open(
            os.path.join(self.config.path.train.score, f"metrics_{self.train_id}.csv"),
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "loss",
                    "accuracy",
                    "recall",
                    "precision",
                    "f1_score",
                    "phase",
                ]
            )
            for epoch in range(self.config.hyperparameter.epochs):
                (
                    train_loss,
                    train_acc,
                    train_precision,
                    train_recall,
                    train_f1,
                ) = self._run_phase(
                    "train", train_loader, model, criterion, optimizer, device
                )
                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        train_acc,
                        train_recall,
                        train_precision,
                        train_f1,
                        "train",
                    ]
                )
                val_loss, val_acc, val_precision, val_recall, val_f1 = self._run_phase(
                    "val", val_loader, model, criterion, optimizer, device
                )
                writer.writerow(
                    [epoch, val_loss, val_acc, val_recall, val_precision, val_f1, "val"]
                )
                # Save the model states
                self._save_model(model, epoch)
            test_loss, test_acc, test_precision, test_recall, test_f1 = self._run_phase(
                "test", test_loader, model, criterion, optimizer, device
            )
            writer.writerow(
                [0, test_loss, test_acc, test_recall, test_precision, test_f1, "test"]
            )

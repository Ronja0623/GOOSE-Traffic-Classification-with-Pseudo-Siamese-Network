import os

import pandas as pd


class DataLoader:
    def __init__(self, config, identifier):
        """
        Initialize the data loader
        """
        self.config = config
        self.identifier = identifier

    def _get_label_structure(self):
        """
        Get the label structure
        """
        label_structure = {}
        for dir in os.listdir(self.config.path.input.json_pcap):
            label_structure[dir] = []
            for subdir in os.listdir(
                os.path.join(self.config.path.input.json_pcap, dir)
            ):
                label_structure[dir].append(subdir)
        return label_structure

    def _get_data_based_on_label(self, label_structure):
        """
        Get the data based on labels
        """
        label_data = {}
        csv_path = os.path.join(
            self.config.path.intermediate.csv, "{label}_{sublabel}_AS1.csv"
        )
        for label, sublabels in label_structure.items():
            combined_df = pd.DataFrame()
            for sublabel in sublabels:
                df = pd.read_csv(csv_path.format(label=label, sublabel=sublabel))
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            label_data[label] = combined_df
        return label_data

    def _check_image_exit(self, df):
        """
        Check if the image exists
        """
        for name in df["Name"]:
            if not os.path.exists(
                os.path.join(self.config.path.intermediate.graph, f"{name}.png")
            ):
                df.drop(df[df["Name"] == name].index, inplace=True)
        return df

    def run(self):
        """
        Run the data loader
        """
        label_structure = self._get_label_structure()
        if self.config.training_dataset.prepared:
            return label_structure, pd.read_csv(
                self.config.training_dataset.selected_file_path
            )
        label_data = self._get_data_based_on_label(label_structure)
        # Calculate the total number of samples to extract per label
        num_samples_per_label = int(
            self.config.training_dataset.data_size / len(label_structure)
        )
        sampled_data = []
        for label, df in label_data.items():
            if len(df) > num_samples_per_label:
                sampled_df = df.sample(
                    n=num_samples_per_label, replace=False, random_state=42
                )
            else:
                sampled_df = df
            sampled_data.append(sampled_df)
        # Combine all sampled DataFrames into a single DataFrame
        combined_df = pd.concat(sampled_data, ignore_index=True)
        # Check if the image exists
        checked_df = self._check_image_exit(combined_df)
        # Save selected data
        checked_df.to_csv(
            os.path.join(
                self.config.path.train.description,
                f"selected_data_{self.identifier}.csv",
            ),
            index=False,
        )
        return label_structure, combined_df

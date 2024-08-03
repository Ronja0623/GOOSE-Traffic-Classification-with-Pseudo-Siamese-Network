import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def remove_substring(main_str, sub_str):
    if sub_str in main_str:
        return main_str.replace(sub_str, "", 1)
    return main_str


def get_sha256_hash(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode("utf-8"))
    return sha256_hash.hexdigest()


def repeat_and_trim(byte_array, required_byte_size):
    repeat_count = (required_byte_size // len(byte_array)) + 1
    byte_array_repeated = np.tile(byte_array, repeat_count)
    return byte_array_repeated[:required_byte_size]


class Preprocess:
    def __init__(
        self,
        config,
    ):
        """
        Initialize the preprocess
        """
        self.config = config

    def _get_data_from_json(self, packet):
        layers = packet["_source"]["layers"]

        frame = layers.get("frame", {})
        # eth = layers.get("eth", {})
        goose = layers.get("goose", {})
        goose_pdu = goose.get("goose.goosePdu_element", {})

        frame_raw = (
            layers.get("frame_raw", [])[0] if layers.get("frame_raw", []) else ""
        )
        eth_raw = layers.get("eth_raw", [])[0] if layers.get("eth_raw", []) else ""
        hex_string = remove_substring(frame_raw, eth_raw)

        new_name = get_sha256_hash(hex_string)

        packet_info = {
            "Name": new_name,
            # 'Time': frame.get("frame.time", ""),
            # 'Source': eth.get("eth.src", ""),
            # 'Destination': eth.get("eth.dst", ""),
            """"Protocols": frame.get("frame.protocols", "").split(":")[-1]
            if "frame.protocols" in frame
            else "","""
            "Length": frame.get("frame.len", ""),
            "Interface_id": frame.get("frame.interface_id", ""),
            "Time_delta": frame.get("frame.time_delta", ""),
            "Coloring_Rule": frame.get("frame.coloring_rule.name", ""),
            "Goose_stNum": goose_pdu.get("goose.stNum", ""),
            "Goose_sqNum": goose_pdu.get("goose.sqNum", ""),
            "Goose_simulation": goose_pdu.get("goose.simulation", ""),
            "Goose_confRev": goose_pdu.get("goose.confRev", ""),
            "Goose_ndsCom": goose_pdu.get("goose.ndsCom", ""),
            "Label": "",
            "Sublabel": "",
        }
        return new_name, packet_info, hex_string

    def _turn_data_to_graph(self, data):
        """
        Turn data to graph.
        """
        byte_array = np.frombuffer(bytes.fromhex(data), dtype=np.uint8)
        image_size = self.config.hyperparameter.model.CNN.required_image_size
        required_byte_size = image_size * image_size * 3
        if len(byte_array) < required_byte_size:
            byte_array = repeat_and_trim(byte_array, required_byte_size)
        else:
            byte_array = byte_array[:required_byte_size]
        return byte_array.reshape(image_size, image_size, 3)

    def run(self):
        """
        Run the preprocess.
        """
        for dir in os.listdir(self.config.path.input.json_pcap):
            for subdir in os.listdir(
                os.path.join(self.config.path.input.json_pcap, dir)
            ):
                label = dir
                sublabel = subdir
                print(f"Processing {label}/{sublabel}...")
                subdir_path = os.path.join(
                    self.config.path.input.json_pcap, dir, subdir
                )
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8") as json_data:
                            packets = json.load(json_data)
                            features = []
                            for packet in tqdm(packets):
                                try:
                                    new_name, feature, data = self._get_data_from_json(
                                        packet
                                    )
                                    feature["Label"] = label
                                    feature["Sublabel"] = sublabel
                                    features.append(feature)
                                    graph = self._turn_data_to_graph(data)
                                    graph_path = os.path.join(
                                        self.config.path.intermediate.graph,
                                        new_name + ".png",
                                    )
                                    plt.imsave(graph_path, graph)
                                except Exception as e:
                                    print(f"Error: {e}")
                                    continue
                            df = pd.DataFrame(features)
                            df.to_csv(
                                os.path.join(
                                    self.config.path.intermediate.csv,
                                    f"{label}_{sublabel}_{os.path.splitext(file_name)[0]}.csv",
                                ),
                                index=False,
                            )

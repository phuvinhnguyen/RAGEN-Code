from dataclasses import dataclass

@dataclass
class SWEEnvConfig:
    huggingface_dataset_name: str = "swebench/swebench-verified"
    split: str = "train"
    sif_path: str = "/swebench/swebench-verified/swebench-verified.sif"
    base_tools_path: str = "/swebench/swebench-verified/base_tools"
    tool_list: list[str] = None
from .app import ContainerEnv
from .config import SWEEnvConfig
from datasets import load_dataset
import re
import os
from difflib import SequenceMatcher

'''
SWE-bench/SWE-smith
SWE-bench/SWE-bench_Verified
SWE-bench/SWE-bench_Lite
SWE-bench/SWE-bench
'''

def distance_to_patch(command, output, location, diff, project_info):
    def normalize_python_code(code: str) -> list:
        lines = code.strip().splitlines()
        norm = []
        for line in lines:
            line = re.sub(r'#.*', '', line)
            line = line.strip()
            if line:
                norm.append(line)
        return norm

    def patch_similarity(diff1: str, diff2: str) -> float:
        norm1 = normalize_python_code(diff1)
        norm2 = normalize_python_code(diff2)
        return SequenceMatcher(None, norm1, norm2).ratio()

    def steps_between(current: str, target: str) -> int:
        rel = os.path.relpath(target, start=current)
        return len([p for p in rel.split(os.sep) if p not in ('.', '')])

    def extract_all_patch_files(patch: str) -> list:
        files = []
        for line in patch.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                path = line[4:].strip()
                # skip /dev/null or empty
                if path == "/dev/null" or not path:
                    continue
                # remove "a/" or "b/"
                path = re.sub(r'^[ab]/', '', path)
                files.append(path)
        return list(set(files))  # unique files

    # -------------------
    patch_text = project_info.get("patch", "")
    if not patch_text:
        return {"score": 0.0, "steps_to_patch_file": 0, "patch_similarity": 0.0}

    # project name
    project_name = project_info['url'].split('/')[-1].split('.')[0]
    base_dir = os.path.join("/tmp", project_name)

    # extract all patched files
    patch_files = extract_all_patch_files(patch_text)
    if not patch_files:
        return {"score": 0.0, "steps_to_patch_file": 0, "patch_similarity": 0.0}

    # total distance (closer => higher score)
    total_inverse_steps = 0.0
    total_steps = 0
    for rel_path in patch_files:
        full_path = os.path.join(base_dir, rel_path)
        steps = steps_between(location, full_path)
        total_steps += steps
        total_inverse_steps += 3 / (1 + steps)  # để tránh chia 0

    # similarity score
    similarity = patch_similarity(diff, patch_text) if diff else 0.0

    # final score: weighted sum (can tune weights)
    final_score = total_inverse_steps + similarity * 30

    return final_score  # cao hơn là tốt hơn

class SWEEnv(ContainerEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SWEEnvConfig()

        self.dataset = load_dataset(self.config.huggingface_dataset_name, split=self.config.split)
        self.dataset = self.dataset.map(lambda x: {
            'problem_statement': x['problem_statement'],
            'hint': '' if 'hint' not in x else x['hint'],
            'base_commit': x['base_commit'],
            'patch': x['patch'],
            'url': f'https://github.com/{x["repo"]}.git'
        })

        super().__init__(
            github_projects=self.dataset,
            sif_path=self.config.sif_path,
            base_tools_path=self.config.base_tools_path,
            scoring_fn=distance_to_patch,
            tool_list=self.config.tool_list,
        )
import subprocess, atexit, random, os
from ragen.env.base import BaseLanguageBasedEnv

class ContainerEnv(BaseLanguageBasedEnv):
    '''GitHub projects is a list of dictionaries, each containing:
        - url: str, the URL of the GitHub project
        - base_commit: str, the commit hash of the base commit
        - problem_statement: str, the problem statement of the project
        - hint: str, the hint of the project or empty string if not provided
        - patch: str, the patch of the project or empty string if not provided
    '''
    def __init__(self,
                 github_projects,
                 sif_path,
                 base_tools_path,
                 scoring_fn = lambda *args, **kwargs: random.random(),
                 tool_list = []
                 ):
        self.github_projects = github_projects
        self.scoring_fn = scoring_fn
        self.base_tools_path = base_tools_path
        self.tool_list = tool_list
        self.sif_path = sif_path
        self.trajectory = []

        self.reset()

        atexit.register(self.close)

    def _run_command(self, command: str):
        """Gửi lệnh vào container và đọc kết quả"""
        if self.proc.poll() is not None:
            raise RuntimeError("Container process is no longer running.")

        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()

            output = []
            while True:
                line = self.proc.stdout.readline()
                if line.strip() == "__END__" or line == "":
                    break
                output.append(line)
            return "".join(output)
        except Exception as e:
            return f"[ERROR] {e}"
        
    def step(self, command: str) -> tuple[str, float, bool]:
        """Gửi lệnh vào container và đọc kết quả
        Returns:
            output (str): observation
            score (float): reward of action (R)
            done (bool): True if the task is completed, False if not
        """
        output = self._run_command(command)
        location = self._run_command("pwd").strip()
        diff = self.get_patch()
        score = self.scoring_fn(command, output, location, diff, self.project_info)
        self.trajectory.append({
            "command": command,
            "output": output,
            "location_after": location,
            "diff": diff,
            "score": score
        })
        return output, score, score>=10., self.trajectory
    
    def reset(self):
        """Reset container và tạo lại"""
        self.close()
        self.trajectory = []

        self.proc = subprocess.Popen(
            [
                "apptainer", "exec",
                "--containall", "--no-home", "--cleanenv",
                "--pwd", "/tmp",
                "--bind", f"{self.base_tools_path}:/mnt/tools",
                self.sif_path,
                "/bin/bash", "-c", 'while read line; do eval "$line"; echo __END__; done'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # -- Setup project --
        # Choose random project
        project = random.choice(self.github_projects)
        name = project['url'].split('/')[-1].split('.')[0]
        self.project_info = {**project, 'name': name}
        # Clone project
        self.step(f"git clone {project['url']}")
        # Move to project directory
        self.step(f"cd {name}")
        # Checkout base_commit
        if 'base_commit' in project:
            self.step(f"git checkout {project['base_commit']}")
        # Install and add tools to PATH
        for tool in self.tool_list:
            self.step(f"bash /mnt/tools/{tool}/install.sh")
            self.step(f"export PATH=$PATH:/mnt/tools/{tool}/bin")
            for script in os.listdir(f"{self.base_tools_path}/{tool}/bin"):
                self.step(f"chmod +x /mnt/tools/{tool}/bin/{script}")

    def close(self):
        """Kết thúc container và dọn dẹp tài nguyên"""
        if hasattr(self, "proc") and self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def get_patch(self) -> str:
        project_name = self.project_info['url'].split('/')[-1].split('.')[0]
        base_commit = self.project_info['base_commit']
        
        # cd vào thư mục project và lấy patch
        output = self._run_command(f"cd /tmp/{project_name} && git diff {base_commit}")
        
        # quay lại thư mục trước (không cần nếu run_command không thay đổi cwd)
        self._run_command("cd -")
        
        return output

    def reset_current_project(self) -> str:
        project_name = self.project_info['url'].split('/')[-1].split('.')[0]
        base_commit = self.project_info['base_commit']
        
        # 1. cd vào thư mục project
        self._run_command(f"cd /tmp/{project_name}")
        
        # 2. reset về commit gốc
        output = self._run_command(f"cd /tmp/{project_name} && git reset --hard {base_commit}")
        
        # 3. quay lại thư mục ban đầu
        self._run_command("cd -")
        
        return output


    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# Ví dụ sử dụng
if __name__ == "__main__":
    container = ContainerEnv(
        github_projects = [
            {
                "url": "https://github.com/namruthahari/Sample-Git-Repo.git",
            }
        ],
        sif_path = "/home/kat/Desktop/FPTAI/swalittle/env/singularity.sif",
        base_tools_path = "/home/kat/Desktop/FPTAI/swalittle/src/tools",
        tool_list = ["search"]
    )


    print("1. pwd:\n", container.step("pwd"))
    print("2. touch testfile.txt:\n", container.step("touch testfile.txt"))
    print("3. ls:\n", container.step("ls"))

    container.close()
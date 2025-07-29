from pathlib import Path


def find_project_root(marker=".git"):
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


print(find_project_root())

from pathlib import Path

def print_project_structure(dir_path: Path, prefix: str = ""):
    """Print project structure, ignoring .pyc, __pycache__, and tests folder."""
    try:
        contents = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return

    # List of names to completely ignore (folders or files)
    ignore_names = {"__pycache__", "tests"}

    # Filter out ignored items and .pyc files
    filtered_contents = [
        item for item in contents 
        if item.name not in ignore_names and item.suffix != ".pyc"
    ]

    for index, item in enumerate(filtered_contents):
        is_last = (index == len(filtered_contents) - 1)
        connector = "└── " if is_last else "├── "
        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_project_structure(item, prefix + extension)

if __name__ == "__main__":
    # Path.cwd() targets your current terminal path (C:\Users\danie\supportmind-ai\backend)
    target_folder = Path.cwd()
    print(f"📁 {target_folder.name}")
    print_project_structure(target_folder)

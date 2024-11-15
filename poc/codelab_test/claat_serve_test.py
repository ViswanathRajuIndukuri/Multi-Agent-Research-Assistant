import subprocess
import os

def convert_markdown_to_codelab(markdown_file, target_dir="my-multi-tab-codelab"):
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Convert Markdown to Codelab HTML format within the target directory
    subprocess.run(["claat", "export", markdown_file], cwd=target_dir)

    # Serve the Codelab locally within the target directory
    subprocess.run(["claat", "serve"], cwd=target_dir)

# Example usage
convert_markdown_to_codelab("example_codelab.md")

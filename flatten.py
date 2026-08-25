#!/usr/bin/env python3
"""
flatten_code.py

Build a flattened_code.md file containing all source code found in a folder
and its subfolders. Each file is titled with its original path.
"""

import argparse
import codecs
import os
import sys
from pathlib import Path


DEFAULT_EXTENSIONS = {
    # Markdown / documentation
    ".md",
    ".markdown",
    ".mdx",

    # Python
    ".py", ".pyi", ".pyx",

    # Web
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",

    # Systems / compiled languages
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx",
    ".cs", ".java", ".go", ".rs", ".swift", ".kt", ".kts", ".scala",

    # Scripting
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".psd1",
    ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".m", ".mm",

    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env", ".properties",

    # Infra / other
    ".sql", ".proto", ".graphql", ".gql", ".tf", ".hcl",
    ".dockerfile", ".makefile", ".mk",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "out",
    "target",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    ".cache",
    "coverage",
    ".gradle",
    ".mvn",
    "vendor",
    ".tox",
    ".eggs",
}

LANGUAGE_MAP = {
    "md": "markdown",
    "markdown": "markdown",
    "mdx": "markdown",

    "py": "python",
    "pyi": "python",
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "jsx": "jsx",
    "ts": "typescript",
    "tsx": "tsx",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "ps1": "powershell",
    "psm1": "powershell",
    "psd1": "powershell",
    "rb": "ruby",
    "php": "php",
    "pl": "perl",
    "pm": "perl",
    "lua": "lua",
    "r": "r",
    "m": "objectivec",
    "mm": "objectivec",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "hh": "cpp",
    "hxx": "cpp",
    "cs": "csharp",
    "java": "java",
    "go": "go",
    "rs": "rust",
    "swift": "swift",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    "sql": "sql",
    "proto": "protobuf",
    "graphql": "graphql",
    "gql": "graphql",
    "tf": "hcl",
    "hcl": "hcl",
    "dockerfile": "docker",
    "makefile": "makefile",
    "mk": "makefile",
    "xml": "xml",
    "vue": "vue",
    "svelte": "svelte",
}


def normalize_extensions(values):
    """Normalize extension list to lowercase dot-prefixed strings."""
    extensions = set()
    for value in values:
        value = value.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        extensions.add(value)
    return extensions


def choose_fence(text):
    """
    Choose a Markdown code fence longer than any run of backticks in the text.

    This prevents embedded code from accidentally closing the fence.
    """
    longest = 0
    current = 0

    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return "`" * max(3, longest + 1)


def language_for(path):
    """Return a Markdown language hint for syntax highlighting."""
    name = path.name.lower()

    if name == "dockerfile":
        return "docker"

    if name in {"makefile", "gnumakefile"}:
        return "makefile"

    suffix = path.suffix.lower().lstrip(".")
    return LANGUAGE_MAP.get(suffix, suffix)


def decode_bytes(data):
    """
    Decode file contents to text.

    Prefers UTF-8, handles common BOMs, and replaces undecodable bytes.
    """
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace")

    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16", errors="replace")

    return data.decode("utf-8", errors="replace")


def is_probably_binary(data):
    """
    Very simple binary-file heuristic.

    Null bytes near the beginning usually mean the file is binary.
    """
    return b"\x00" in data[:8192]


def iter_source_files(
    root,
    output_path,
    extensions,
    exclude_dirs,
    include_all_files,
    max_size,
):
    """Yield candidate files recursively."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Sort directories for stable output and prune excluded directories.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)

        for filename in sorted(filenames):
            path = Path(dirpath) / filename

            try:
                resolved = path.resolve()
            except OSError:
                continue

            # Do not include the output file itself.
            if resolved == output_path:
                continue

            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            if not include_all_files and path.suffix.lower() not in extensions:
                continue

            try:
                size = path.stat().st_size
            except OSError:
                continue

            if max_size > 0 and size > max_size:
                continue

            yield path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten source code files into a single Markdown file. "
            "Each file is written under a heading containing its path."
        )
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Folder to scan. Defaults to the current directory.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="flattened_code.md",
        help="Output Markdown file. Defaults to flattened_code.md.",
    )

    parser.add_argument(
        "-e",
        "--extensions",
        nargs="*",
        default=None,
        help=(
            "File extensions to include, e.g. .py .js ts. "
            "Defaults to a built-in list of common source-code extensions. "
            "Use --all-files to include all files instead."
        ),
    )

    parser.add_argument(
        "-x",
        "--exclude-dir",
        action="append",
        default=[],
        help=(
            "Additional directory names to exclude. "
            "Can be specified multiple times."
        ),
    )

    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Include all non-binary files, not just known source extensions.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1_000_000,
        help=(
            "Skip files larger than this many bytes. "
            "Use 0 for no limit. Default: 1000000."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.extensions is None:
        extensions = DEFAULT_EXTENSIONS
    else:
        extensions = normalize_extensions(args.extensions)

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs.update(args.exclude_dir)

    output.parent.mkdir(parents=True, exist_ok=True)

    included_count = 0
    skipped_binary_count = 0
    skipped_read_count = 0

    with output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Flattened code\n\n")
        out.write(f"Root: `{root.as_posix()}`\n\n")

        for path in iter_source_files(
            root=root,
            output_path=output,
            extensions=extensions,
            exclude_dirs=exclude_dirs,
            include_all_files=args.all_files,
            max_size=args.max_size,
        ):
            try:
                data = path.read_bytes()
            except OSError as exc:
                print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
                skipped_read_count += 1
                continue

            if is_probably_binary(data):
                skipped_binary_count += 1
                continue

            text = decode_bytes(data)

            # Ensure the fenced block ends cleanly.
            if not text.endswith("\n"):
                text += "\n"

            rel_path = path.relative_to(root).as_posix()
            fence = choose_fence(text)
            language = language_for(path)

            out.write(f"## {rel_path}\n\n")
            out.write(f"{fence}{language}\n")
            out.write(text)
            out.write(f"{fence}\n\n")

            included_count += 1

        if included_count == 0:
            out.write("_No matching files found._\n")

    print(f"Wrote {included_count} file(s) to {output}")

    if skipped_binary_count:
        print(
            f"Skipped {skipped_binary_count} probably-binary file(s).",
            file=sys.stderr,
        )

    if skipped_read_count:
        print(
            f"Skipped {skipped_read_count} unreadable file(s).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
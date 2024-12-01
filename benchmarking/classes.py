from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass 
class IngestionHeuristics:
    min_files_to_consider_dir_for_problems: int
    min_file_content_len: int

@dataclass
class File:
    path: Path
    contents: str

@dataclass
class EmbeddedFile:
    path: str
    contents: str
    embedding: list

    def __str__(self):
        return f"File: {self.path}, Length: {len(self.contents)}"
    
    def __repr__(self) -> str:
        return f"File: {self.path}, Length: {len(self.contents)}"


@dataclass
class FilePair:
    cosine_similarity: float
    files: List[EmbeddedFile]

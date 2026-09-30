from pathlib import Path
from src.document_loader import load_documents
for d in load_documents(Path("documents")):
    print(d.filename, len(d.pages))
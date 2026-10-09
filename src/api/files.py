"""File management: list, upload, delete documents in /documents."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import Config
from src.document_loader import SUPPORTED_EXTENSIONS, is_supported

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def register_file_routes(app: FastAPI, cfg: Config, auth) -> None:
    docs_dir = cfg.documents_path
    docs_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/files/list")
    async def list_files(user=Depends(auth.require_auth)):
        files = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            rel = path.relative_to(docs_dir)
            stat = path.stat()
            files.append({
                "name": str(rel),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "supported": is_supported(path),
            })
        return {"files": files, "total": len(files)}

    @app.post("/files/upload")
    async def upload_file(file: UploadFile = File(...), user=Depends(auth.require_auth)):
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        safe_name = Path(file.filename).name
        if safe_name in {"", ".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid filename")

        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. "
                       f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        target = docs_dir / safe_name
        total = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        out.close()
                        target.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)",
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            logger.exception("Upload failed")
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

        logger.info("Uploaded %s (%d bytes)", safe_name, total)
        return {"name": safe_name, "size": total, "message": "Upload successful"}

    @app.delete("/files/{name:path}")
    async def delete_file(name: str, user=Depends(auth.require_auth)):
        target = (docs_dir / name).resolve()
        if not str(target).startswith(str(docs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {name}")
        try:
            target.unlink()
        except Exception as exc:
            logger.exception("Delete failed")
            raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc
        return {"name": name, "message": "Deleted"}

    @app.get("/files/download/{name:path}")
    async def download_file(name: str, user=Depends(auth.require_auth)):
        target = (docs_dir / name).resolve()
        if not str(target).startswith(str(docs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(target), filename=target.name)
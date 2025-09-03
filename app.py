# app.py — Image↔Text Manifest API (FastAPI)

import os
import tempfile
from typing import Optional

from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, Depends, Header, Response
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from image_to_text_full_v3 import (
    encode_lossless_to_manifest, decode_lossless_manifest_to_image,
    encode_lossy_algo_to_text,  decode_lossy_algo_text_to_image,
    encode_lossy_nlp_to_text,   decode_lossy_nlp_text_to_proxy_image,
)

# =========================
# Auth via X-API-Key header
# =========================
API_KEY = os.environ.get("IMAGE_TEXT_API_KEY", "change-me")

def require_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# =========
# FastAPI app
# =========
app = FastAPI(title="Image↔Text Manifest API")

# CORS (mở, có thể siết lại origins sau)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # ví dụ: ["https://chat.openai.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "Content-Type"],
)

# =============
# Health checks
# =============
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "img2txt", "endpoints": ["/encode/*", "/decode/*"]}

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

# ==================
# Helper: lưu file tạm
# ==================
def _save_upload_to_tmp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.file.read())
        tmp_path = f.name
    return tmp_path

# ============
# LOSSLESS I/O
# ============
class LosslessDecodeIn(BaseModel):
    manifest: str
    output_dir: Optional[str] = "."

@app.post("/encode/lossless", dependencies=[Depends(require_key)])
async def encode_lossless(
    image: UploadFile = File(...),
    source: str = Form("chat_upload"),
):
    tmp = _save_upload_to_tmp(image)
    try:
        manifest, out_name = encode_lossless_to_manifest(tmp, source=source)
        return {"manifest": manifest, "suggested_filename": out_name}
    finally:
        try: os.remove(tmp)
        except Exception: pass

@app.post("/decode/lossless", dependencies=[Depends(require_key)])
def decode_lossless(body: LosslessDecodeIn):
    out_path, info = decode_lossless_manifest_to_image(body.manifest, output_dir=body.output_dir)
    return {"rebuilt_path": out_path, "info": info}

# ===============
# LOSSY-ALGO I/O
# ===============
class LossyAlgoDecodeIn(BaseModel):
    manifest: str
    output_dir: Optional[str] = "."
    out_name: Optional[str] = None

@app.post("/encode/lossy-algo", dependencies=[Depends(require_key)])
async def encode_lossy_algo(
    image: UploadFile = File(...),
    source: str = Form("chat_upload"),
    lock_dims: bool = Form(True),
    max_side: int = Form(128),
    palette_size: int = Form(32),
    resample: str = Form("bicubic"),    # nearest|bilinear|bicubic|lanczos
    dither: bool = Form(False),
):
    tmp = _save_upload_to_tmp(image)
    try:
        text, name = encode_lossy_algo_to_text(
            tmp, source=source, lock_dims=lock_dims,
            max_side=max_side, palette_size=palette_size,
            resample=resample, dither=dither
        )
        return {"manifest": text, "suggested_filename": name}
    finally:
        try: os.remove(tmp)
        except Exception: pass

@app.post("/decode/lossy-algo", dependencies=[Depends(require_key)])
def decode_lossy_algo(body: LossyAlgoDecodeIn):
    out_path, info = decode_lossy_algo_text_to_image(
        body.manifest, output_dir=body.output_dir, out_name=body.out_name
    )
    return {"rebuilt_path": out_path, "info": info}

# =============
# LOSSY-NLP I/O
# =============
class LossyNlpDecodeIn(BaseModel):
    manifest: str
    output_dir: Optional[str] = "."
    out_name: Optional[str] = "rebuilt_lossy_nlp_proxy_v3.png"

@app.post("/encode/lossy-nlp", dependencies=[Depends(require_key)])
async def encode_lossy_nlp(
    image: UploadFile = File(...),
    source: str = Form("chat_upload"),
    preserve_dims: bool = Form(True),
    target_short_side: int = Form(512),
    palette_probe: int = Form(8),
):
    tmp = _save_upload_to_tmp(image)
    try:
        text, name = encode_lossy_nlp_to_text(
            tmp, source=source, preserve_dims=preserve_dims,
            target_short_side=target_short_side, palette_probe=palette_probe
        )
        return {"manifest": text, "suggested_filename": name}
    finally:
        try: os.remove(tmp)
        except Exception: pass

@app.post("/decode/lossy-nlp", dependencies=[Depends(require_key)])
def decode_lossy_nlp(body: LossyNlpDecodeIn):
    out_path, info = decode_lossy_nlp_text_to_proxy_image(
        body.manifest, output_dir=body.output_dir, out_name=body.out_name
    )
    return {"rebuilt_path": out_path, "info": info}

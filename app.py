"""
Golden Fade — AI Style Preview backend
----------------------------------------
Wraps the open-source, MIT-licensed HairFastGAN model
(https://github.com/AIRI-Institute/HairFastGAN, NeurIPS 2024) so the
Golden Fade website can send a customer's photo + a chosen reference
hairstyle photo, and get back a real generated preview image —
no paid AI API involved.

How it works:
  1. A public Hugging Face Space runs the model on a shared GPU, for
     free, under an MIT license.
  2. This backend is a small, free-to-host proxy: it receives an
     upload from our website, forwards it to that Space using the
     official `gradio_client` library, waits for the result, and
     returns the generated image back to the browser.
  3. This proxy itself needs no GPU — it can run on a free CPU-only
     host (e.g. Render's free tier). See README.md for deploy steps.

IMPORTANT — why this points at a different Space than before:
  This originally called the official AIRI-Institute/HairFastGAN
  Space directly. That Space's own API endpoint has a known,
  long-standing bug: calling it via gradio_client raises
      "The upstream Gradio app has raised an exception but has not
       enabled verbose error reporting."
  for everyone, confirmed by multiple independent reports on its own
  Hugging Face discussion page since mid-2024, with no fix posted.
  It is not something fixable from our side.

  This version instead calls "multimodalart/hairfastgan" — a
  separate, independently-run public Space wrapping the same
  underlying model with a simpler interface (no blending/poisson
  tuning options, just three image inputs).

IMPORTANT — one thing I could not verify from this environment:
  I don't have live network access to actually call this Space and
  confirm the request succeeds end-to-end. The parameter names below
  (source / target_1 / target_2, api_name="/swap_hair") are inferred
  from that Space's own published app.py source code — its click
  handler is `btn.click(fn=swap_hair, inputs=[source, target_1,
  target_2], ...)`, and Gradio names an endpoint after its handler
  function by default, so "/swap_hair" is the most likely name, but
  Hugging Face Spaces can change their code over time. If this
  returns an error after you deploy it, run:

      from gradio_client import Client
      Client("multimodalart/hairfastgan").view_api()

  from any machine with internet access — it prints the exact current
  parameter names, and you (or I, in a future session) can update the
  `client.predict(...)` call below to match.
"""

import os
import tempfile
import shutil
import logging
import urllib.request

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from gradio_client import Client, file as gr_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("golden-fade-hairfast")

app = FastAPI(title="Golden Fade AI Style Preview")

# Allow the barbershop website (any origin, since this is a small
# public demo endpoint with no sensitive data) to call this API from
# browser JavaScript.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The underlying open-source model demo, hosted for free by a
# third party. Swap this for your own duplicated Space (see README)
# if you outgrow the shared public queue, or if this mirror ever
# goes down too.
HF_SPACE = os.environ.get("HAIRFAST_SPACE", "multimodalart/hairfastgan")

_client = None


def get_client() -> Client:
    """Lazily connect to the Hugging Face Space (connecting at import
    time would slow down / break the server's startup if HF is briefly
    unreachable)."""
    global _client
    if _client is None:
        log.info("Connecting to Hugging Face Space: %s", HF_SPACE)
        _client = Client(HF_SPACE)
    return _client


@app.get("/")
def health():
    return {"status": "ok", "model": HF_SPACE}


@app.post("/api/preview")
async def generate_preview(
    face: UploadFile = File(..., description="Customer's photo"),
    shape: UploadFile | None = File(None, description="Reference photo for the desired hairstyle shape (upload)"),
    color: UploadFile | None = File(None, description="Optional separate reference for hair colour (upload)"),
    shape_url: str | None = Form(None, description="Alternative to uploading 'shape': a direct image URL, e.g. one of Golden Fade's own gallery photos"),
    color_url: str | None = Form(None, description="Alternative to uploading 'color': a direct image URL"),
):
    """
    Accepts a customer's face photo, plus a reference hairstyle either
    as an uploaded file (shape/color) or as a direct image URL
    (shape_url/color_url) — the website uses shape_url so it can reuse
    the same photos already shown in the gallery, without asking the
    customer to download and re-upload them.
    """
    if shape is None and not shape_url:
        raise HTTPException(status_code=400, detail="Provide either 'shape' (file) or 'shape_url'.")

    tmp_dir = tempfile.mkdtemp(prefix="goldenfade_")
    try:
        face_path = os.path.join(tmp_dir, "face_" + face.filename)
        with open(face_path, "wb") as f:
            shutil.copyfileobj(face.file, f)

        if shape is not None:
            shape_path = os.path.join(tmp_dir, "shape_" + shape.filename)
            with open(shape_path, "wb") as f:
                shutil.copyfileobj(shape.file, f)
        else:
            shape_path = os.path.join(tmp_dir, "shape_from_url.jpg")
            urllib.request.urlretrieve(shape_url, shape_path)

        if color is not None:
            color_path = os.path.join(tmp_dir, "color_" + color.filename)
            with open(color_path, "wb") as f:
                shutil.copyfileobj(color.file, f)
        elif color_url:
            color_path = os.path.join(tmp_dir, "color_from_url.jpg")
            urllib.request.urlretrieve(color_url, color_path)
        else:
            color_path = shape_path  # reuse the shape photo for colour too

        client = get_client()

        try:
            result = client.predict(
                source=gr_file(face_path),
                target_1=gr_file(shape_path),
                target_2=gr_file(color_path),
                api_name="/swap_hair",
            )
        except Exception as e:
            log.exception("HairFastGAN call failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "The underlying AI model didn't respond as expected. "
                    "This usually means the Space's API changed — run "
                    f"Client('{HF_SPACE}').view_api() to see "
                    "the current parameter names and update app.py. "
                    f"Original error: {e}"
                ),
            )

        result_path = result[0] if isinstance(result, (list, tuple)) else result

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=502, detail="Model returned no image.")

        return FileResponse(result_path, media_type="image/png")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

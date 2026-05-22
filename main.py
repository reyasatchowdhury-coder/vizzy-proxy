import os, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ATLAS_KEY  = os.environ.get("ATLASCLOUD_API_KEY", "")
ATLAS_BASE = "https://api.atlascloud.ai"

class VideoReq(BaseModel):
    video_prompt: str
    platform: str = "TikTok"
    duration: int = 5
    resolution: str = "720p"
    model: str = "bytedance/seedance-2.0/text-to-video"

@app.get("/health")
def health():
    return {"ok": True, "key_set": bool(ATLAS_KEY)}

@app.post("/generate-video")
async def generate_video(req: VideoReq):
    if not ATLAS_KEY:
        raise HTTPException(status_code=500, detail="ATLASCLOUD_API_KEY not set in environment.")
    aspect = "9:16" if req.platform in ("TikTok","Instagram Reels","YouTube Shorts") else "16:9"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{ATLAS_BASE}/api/v1/model/generateImage",
            headers={"Authorization": f"Bearer {ATLAS_KEY}", "Content-Type": "application/json"},
            json={"model": req.model, "prompt": req.video_prompt,
                  "duration": req.duration, "resolution": req.resolution, "aspect_ratio": aspect})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    d = r.json()
    rid = d.get("request_id") or d.get("id") or (d.get("data") or {}).get("request_id")
    if not rid:
        raise HTTPException(status_code=500, detail=f"No request_id in response: {d}")
    return {"request_id": rid}

@app.get("/poll/{request_id}")
async def poll(request_id: str):
    if not ATLAS_KEY:
        raise HTTPException(status_code=500, detail="ATLASCLOUD_API_KEY not set.")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{ATLAS_BASE}/api/v1/model/result/{request_id}",
            headers={"Authorization": f"Bearer {ATLAS_KEY}"})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    d = r.json()
    raw = ((d.get("status") or (d.get("data") or {}).get("status")) or "pending").lower()
    status = ("success" if raw in ("success","succeeded","completed","done")
              else "failed" if raw in ("failed","error","cancelled")
              else "processing")
    url = ((d.get("output") or {}).get("url") or (d.get("data") or {}).get("url")
           or d.get("url") or (d.get("result") or {}).get("url"))
    return {"status": status, "url": url}

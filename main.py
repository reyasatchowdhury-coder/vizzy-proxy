import os, httpx, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ATLAS_KEY   = os.environ.get("ATLASCLOUD_API_KEY", "")
OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
ATLAS_BASE  = "https://api.atlascloud.ai"
OPENAI_BASE = "https://api.openai.com/v1"

SCRIPT_PROMPT = """You are an expert video ad creative director. Return ONLY a JSON object — no markdown, no explanation, no code fences.
{
  "product_summary":"1-2 sentence summary",
  "target_audience":"specific audience",
  "hooks":[
    {"style":"Problem-agitate","text":"hook line"},
    {"style":"Bold claim","text":"hook line"},
    {"style":"Curiosity gap","text":"hook line"}
  ],
  "scripts":[
    {"variation":"name","angle":"angle","tone":"tone","duration":"15s","script":"[VISUAL: scene]\\nSpoken: line\\n[VISUAL: scene]\\nSpoken: line","cta":"CTA text","video_prompt":"Vivid 2-3 sentence AI video generation prompt describing visual style, setting, motion and mood"},
    {"variation":"name","angle":"angle","tone":"tone","duration":"15s","script":"...","cta":"...","video_prompt":"..."},
    {"variation":"name","angle":"angle","tone":"tone","duration":"30s","script":"...","cta":"...","video_prompt":"..."}
  ],
  "avatar_style":"description","music_mood":"description","caption_style":"description","best_posting_time":"description"
}"""

class ScriptReq(BaseModel):
    product: str
    platform: str = "TikTok"
    objective: str = "Drive purchases"

class VideoReq(BaseModel):
    video_prompt: str
    platform: str = "TikTok"
    duration: int = 5
    resolution: str = "720p"
    model: str = "bytedance/seedance-2.0/text-to-video"

@app.get("/health")
def health():
    return {"ok": True, "atlas_key": bool(ATLAS_KEY), "openai_key": bool(OPENAI_KEY)}

@app.post("/generate-script")
async def generate_script(req: ScriptReq):
    if not OPENAI_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in Railway environment.")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": SCRIPT_PROMPT},
                    {"role": "user", "content": f"Product: {req.product}\nPlatform: {req.platform}\nObjective: {req.objective}"}
                ],
                "temperature": 0.7
            })
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    text = r.json()["choices"][0]["message"]["content"].strip().replace("```json","").replace("```","").strip()
    try:
        return {"brief": json.loads(text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)} — raw: {text[:300]}")

@app.post("/generate-video")
async def generate_video(req: VideoReq):
    if not ATLAS_KEY:
        raise HTTPException(status_code=500, detail="ATLASCLOUD_API_KEY not set in Railway environment.")
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
        raise HTTPException(status_code=500, detail=f"No request_id: {d}")
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

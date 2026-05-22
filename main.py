import os, httpx, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ATLAS_KEY  = os.environ.get("ATLASCLOUD_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ATLAS_BASE = "https://api.atlascloud.ai"

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
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.openai.com/v1/chat/completions",
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
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")

@app.post("/generate-video")
async def generate_video(req: VideoReq):
    if not ATLAS_KEY:
        raise HTTPException(status_code=500, detail="ATLASCLOUD_API_KEY not set.")
    aspect = "9:16" if req.platform in ("TikTok","Instagram Reels","YouTube Shorts") else "16:9"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{ATLAS_BASE}/api/v1/model/generateImage",
            headers={"Authorization": f"Bearer {ATLAS_KEY}", "Content-Type": "application/json"},
            json={"model": req.model, "prompt": req.video_prompt,
                  "duration": req.duration, "resolution": req.resolution, "aspect_ratio": aspect})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    d = r.json()

    # Atlas Cloud returns id inside 'data' and a 'get' poll URL
    data = d.get("data", {})
    rid = data.get("id") or d.get("id") or d.get("request_id")
    poll_url = data.get("get") or d.get("get")

    if not rid and not poll_url:
        raise HTTPException(status_code=500, detail=f"No job ID in response: {d}")

    return {"request_id": rid, "poll_url": poll_url}

@app.get("/poll/{request_id}")
async def poll(request_id: str):
    if not ATLAS_KEY:
        raise HTTPException(status_code=500, detail="ATLASCLOUD_API_KEY not set.")
    # Use the correct Atlas Cloud prediction endpoint
    poll_url = f"{ATLAS_BASE}/api/v1/model/prediction/{request_id}"
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(poll_url, headers={"Authorization": f"Bearer {ATLAS_KEY}"})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    d = r.json()
    data = d.get("data", d)
    raw = (data.get("status") or d.get("status") or "processing").lower()
    status = ("success" if raw in ("success","succeeded","completed","done")
              else "failed" if raw in ("failed","error","cancelled")
              else "processing")
    # Extract video URL from outputs
    outputs = data.get("outputs") or d.get("outputs")
    url = None
    if outputs:
        if isinstance(outputs, list) and len(outputs) > 0:
            url = outputs[0]
        elif isinstance(outputs, dict):
            url = outputs.get("url") or outputs.get("video")
    if not url:
        url = data.get("url") or d.get("url")
    return {"status": status, "url": url, "raw_status": raw}

@app.post("/poll-by-url")
async def poll_by_url(body: dict):
    """Poll using the exact URL returned by Atlas Cloud"""
    poll_url = body.get("poll_url")
    if not poll_url:
        raise HTTPException(status_code=400, detail="poll_url required")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(poll_url, headers={"Authorization": f"Bearer {ATLAS_KEY}"})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    d = r.json()
    data = d.get("data", d)
    raw = (data.get("status") or d.get("status") or "processing").lower()
    status = ("success" if raw in ("success","succeeded","completed","done")
              else "failed" if raw in ("failed","error","cancelled")
              else "processing")
    outputs = data.get("outputs") or d.get("outputs")
    url = None
    if outputs:
        if isinstance(outputs, list) and len(outputs) > 0:
            url = outputs[0]
        elif isinstance(outputs, dict):
            url = outputs.get("url") or outputs.get("video")
    if not url:
        url = data.get("url") or d.get("url")
    return {"status": status, "url": url, "raw": d}

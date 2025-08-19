# main.py
import os, json, time, base64
from datetime import datetime, timezone, timedelta
import requests, math
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from playwright.sync_api import sync_playwright

# ---------------- config from env / secrets ----------------
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT  = os.environ.get("TG_CHAT")
DG_URLS = [ os.environ.get("DG_URL1","https://dg18.co/wap/"), os.environ.get("DG_URL2","https://dg18.co/") ]
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW","3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE","6"))
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY")  # owner/repo
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
STATE_PATH = "state.json"
# Malaysia timezone
TZ = timezone(timedelta(hours=8))

def log(msg):
    print(f"[DGMON] {datetime.now(TZ).isoformat()} - {msg}")

# GitHub state helpers (store state.json via contents API)
def github_get_state():
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return {"sha":None, "data":{}}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        j = r.json()
        try:
            return {"sha": j["sha"], "data": json.loads(base64.b64decode(j["content"]).decode())}
        except:
            return {"sha": j["sha"], "data": {}}
    else:
        return {"sha": None, "data": {}}

def github_put_state(data, sha=None, message="update state"):
    if not GITHUB_REPO or not GITHUB_TOKEN:
        log("GITHUB not configured. Skipping save.")
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": os.environ.get("GITHUB_REF","main")
    }
    if sha:
        payload["sha"]=sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200,201):
        log("Saved state.json")
        return True
    else:
        log(f"Save state failed: {r.status_code} {r.text}")
        return False

# Telegram send
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG_TOKEN or TG_CHAT missing, cannot send.")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code == 200:
            log("Telegram sent.")
            return True
        else:
            log(f"Telegram error: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram exception: {e}")
        return False

# image analysis: detect red/blue blobs and infer runs (简化 & 稳健)
def analyze_image_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h,w = np_img.shape[:2]
    scale = 1.0
    if max(h,w) > 1600:
        scale = 1600/max(h,w)
        np_img = cv2.resize(np_img, (int(w*scale), int(h*scale)))
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    # blue range
    mask_b = cv2.inRange(hsv, (90,60,60), (140,255,255))
    # red range (two ranges)
    mask_r1 = cv2.inRange(hsv, (0,60,60), (10,255,255))
    mask_r2 = cv2.inRange(hsv, (170,60,60), (180,255,255))
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask_any = cv2.bitwise_or(mask_b, mask_r)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_any>0).astype("uint8")*255)
    blobs=[]
    for i in range(1,num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_SIZE: continue
        cx = int(centroids[i][0]); cy=int(centroids[i][1])
        c = mask_b[cy,cx]>0
        color = 'P' if c else ('B' if mask_r[cy,cx]>0 else 'U')
        blobs.append({"x":cx,"y":cy,"area":int(area),"color":color})
    if len(blobs)==0:
        return {"boards":[], "summary":{"total_blobs":0}}
    # cluster blobs spatially (scikit-learn DBSCAN)
    from sklearn.cluster import DBSCAN
    pts = np.array([[b["x"], b["y"]] for b in blobs])
    db = DBSCAN(eps=120*scale, min_samples=1).fit(pts)
    labels_db = db.labels_
    clusters={}
    for i,lb in enumerate(labels_db):
        clusters.setdefault(int(lb), []).append(blobs[i])
    boards=[]
    long_count=0; super_long_count=0; longish_count=0
    for cid, items in clusters.items():
        items_sorted = sorted(items, key=lambda b:(b["x"], b["y"]))
        # group by approximate x columns
        cols=[]
        for it in items_sorted:
            if not cols or abs(it["x"]-cols[-1][-1]["x"])>24*scale:
                cols.append([it])
            else:
                cols[-1].append(it)
        seq=[]
        for col in cols:
            col_sorted = sorted(col, key=lambda b:b["y"])
            seq.extend([p["color"] for p in col_sorted])
        # runs
        runs=[]
        if seq:
            cur = seq[0]; ln=1
            for s in seq[1:]:
                if s==cur: ln+=1
                else:
                    runs.append((cur,ln)); cur=s; ln=1
            runs.append((cur,ln))
        max_run = max([r[1] for r in runs]) if runs else 0
        cat='other'
        if max_run>=10:
            cat='super_long'; super_long_count+=1; long_count+=1
        elif max_run>=8:
            cat='long'; long_count+=1
        elif max_run>=4:
            cat='longish'; longish_count+=1
        boards.append({"cluster":cid,"count":len(items),"max_run":max_run,"category":cat,"runs":runs})
    summary={"total_blobs":len(blobs),"board_clusters":len(boards),"long_count":long_count,"super_long_count":super_long_count,"longish_count":longish_count}
    return {"boards":boards,"summary":summary}

# helper: try click "Free" and attempt slider
def attempt_enter_and_slider(page):
    texts = ["Free","Free Play","免费试玩","免费","试玩"]
    for t in texts:
        try:
            el = page.query_selector(f"text={t}")
            if el:
                try:
                    el.click(timeout=3000)
                    page.wait_for_timeout(1500)
                    return True
                except:
                    pass
        except: pass
    # try generic button
    try:
        btn = page.query_selector("button")
        if btn:
            btn.click(); page.wait_for_timeout(1000); return True
    except: pass
    # try slider drag
    try:
        el = page.query_selector("[role=slider], .slider, .drag, .verify-slider, .slide-btn")
        if el:
            box = el.bounding_box()
            if box:
                sx = box["x"]+2; sy = box["y"] + box["height"]/2
                ex = box["x"] + box["width"] - 4
                page.mouse.move(sx, sy); page.mouse.down()
                page.mouse.move(ex, sy, steps=20); page.mouse.up()
                page.wait_for_timeout(1000)
                return True
    except: pass
    return False

def capture_and_analyze(page):
    img_bytes = page.screenshot(full_page=True)
    return analyze_image_bytes(img_bytes), img_bytes

def main():
    state_obj = github_get_state()
    sha = state_obj.get("sha")
    state = state_obj.get("data") or {}
    active = state.get("active", False)
    active_since = state.get("active_since")
    cooldown_until = int(state.get("cooldown_until", 0) or 0)
    now_ts = int(time.time()*1000)
    log(f"Run start. prev active={active} since={active_since} cooldown_until={cooldown_until}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1280,"height":800})
        page = context.new_page()
        opened=False
        for url in DG_URLS:
            try:
                log(f"Opening {url}")
                page.goto(url, timeout=30000)
                page.wait_for_timeout(1500)
                attempt_enter_and_slider(page)
                page.wait_for_timeout(2000)
                opened=True
                break
            except Exception as e:
                log(f"Open error {url}: {e}")
                continue
        if not opened:
            log("Could not open DG pages.")
            browser.close(); return

        analysis, img_bytes = capture_and_analyze(page)
        log(f"Analysis summary: {analysis.get('summary')}")
        s = analysis.get("summary",{})
        long_count = s.get("long_count",0)
        super_long = s.get("super_long_count",0)
        longish = s.get("longish_count",0)
        total_clusters = s.get("board_clusters",0)
        overall = "胜率中等（平台收割中等时段）"
        if long_count >= MIN_BOARDS_FOR_POW:
            overall = "放水时段（提高胜率）"
        elif long_count >= MID_LONG_REQ and longish>0:
            overall = "中等胜率（中上）"
        else:
            sparse = sum(1 for b in analysis.get("boards",[]) if b["count"]<6)
            if total_clusters>0 and sparse >= total_clusters*0.6:
                overall = "胜率调低（平台收割时段）"
            else:
                overall = "胜率中等（平台收割中等时段）"

        log(f"Overall -> {overall}")

        now_iso = datetime.now(TZ).isoformat()
        in_cooldown = now_ts < cooldown_until

        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            if active:
                # update last_seen
                state.update({"active":True,"active_since":active_since,"last_seen":now_iso,"cooldown_until":0})
                github_put_state(state, sha=sha, message="update active seen")
                log("Still active; no new notify.")
            else:
                if in_cooldown:
                    log("In cooldown; skipping notify.")
                else:
                    text = f"[DG提醒] 发现放水/中上局势\n判定: {overall}\n长龙(>=8) 桌数: {long_count}\n超长龙(>=10): {super_long}\n时间: {now_iso}\n备注: 按设阈值触发。"
                    send_telegram(text)
                    # mark active
                    state = {"active":True,"active_since":now_iso,"last_seen":now_iso,"cooldown_until":0}
                    github_put_state(state, sha=sha, message="start active")
        else:
            if active:
                # ended -> compute duration
                try:
                    start_dt = datetime.fromisoformat(active_since)
                    end_dt = datetime.now(TZ)
                    dur_min = int((end_dt - start_dt).total_seconds()/60)
                    text = f"[DG提醒] 放水已结束\n开始: {active_since}\n结束: {end_dt.isoformat()}\n共持续: {dur_min} 分钟"
                    send_telegram(text)
                except Exception as e:
                    log(f"Duration compute error: {e}")
                # set cooldown
                cooldown_ms = COOLDOWN_MINUTES*60*1000
                new_state = {"active":False,"active_since":None,"last_seen":now_iso,"cooldown_until": int(time.time()*1000)+cooldown_ms}
                github_put_state(new_state, sha=sha, message="end active")
            else:
                log("No active, nothing to do.")

        browser.close()

if __name__ == "__main__":
    main()

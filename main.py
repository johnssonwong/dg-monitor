# main.py
# DG Monitor - 完整版（含时间格式、预计持续时长估算、历史记录）
import os, json, time, base64, traceback, sys
from datetime import datetime, timezone, timedelta
from io import BytesIO

# libs
try:
    import requests
    import numpy as np
    import cv2
    from PIL import Image
    from playwright.sync_api import sync_playwright
    SKLEARN_AVAILABLE = True
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        SKLEARN_AVAILABLE = False
except Exception as e:
    requests = None; np = None; cv2 = None; Image = None; sync_playwright = None; DBSCAN = None; SKLEARN_AVAILABLE = False

# ---------------- Config (from env / secrets) ----------------
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT  = os.environ.get("TG_CHAT")
DG_URLS = [ os.environ.get("DG_URL1", "https://dg18.co/wap/"), os.environ.get("DG_URL2", "https://dg18.co/") ]
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW", "3"))   # 放水判定：至少多少张长龙/超长龙
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))              # 中等(中上)门槛（长龙基础）
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))     # 结束后冷却
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE", "6"))
CAPTURE_SAMPLES = int(os.environ.get("CAPTURE_SAMPLES", "5"))
STATE_PATH = "state.json"
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY") or os.environ.get("GITHUB_REPO")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ESTIMATED_DEFAULT_MINUTES = int(os.environ.get("ESTIMATED_DEFAULT_MINUTES", "10"))
# Malaysia timezone
TZ = timezone(timedelta(hours=8))
# -------------------------------------------------------------

def now_dt():
    return datetime.now(TZ)

def format_dt(dt):
    # format like: 2025-08-23 07:50:00 (UTC+8)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " (UTC+8)"

def minutes_between(dt_start, dt_end):
    return int((dt_end - dt_start).total_seconds() // 60)

def log(msg):
    print(f"[DGMON] {format_dt(now_dt())} - {msg}")

# ---------- GitHub state helpers ----------
def github_get_state():
    if not GITHUB_REPO or not GITHUB_TOKEN or requests is None:
        return {"sha": None, "data": {}}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            j = r.json()
            try:
                content = base64.b64decode(j["content"]).decode()
                return {"sha": j["sha"], "data": json.loads(content)}
            except Exception:
                return {"sha": j["sha"], "data": {}}
        else:
            return {"sha": None, "data": {}}
    except Exception as e:
        log(f"GitHub get state error: {e}")
        return {"sha": None, "data": {}}

def github_put_state(data, sha=None, message="update state"):
    if not GITHUB_REPO or not GITHUB_TOKEN or requests is None:
        log("GitHub not configured - cannot save state.")
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": os.environ.get("GITHUB_REF", "main")
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(url, headers=headers, json=payload, timeout=20)
        if r.status_code in (200,201):
            log("Saved state.json to repo.")
            return True
        else:
            log(f"Save state failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Save state exception: {e}")
        return False

# ---------- Telegram ----------
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT or requests is None:
        log("Telegram token/chat missing or requests unavailable.")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TG_CHAT, "text": text}, timeout=20)
        if r.status_code == 200:
            log("Telegram sent.")
            return True
        else:
            log(f"Telegram send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram exception: {e}")
        return False

# ---------- Image processing helpers (morphology & blobs) ----------
def detect_boards_from_mask(mask_any, scale):
    try:
        m = (mask_any > 0).astype('uint8') * 255
        kernel_size = max(5, int(12*scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10*scale), int(4*scale)))
        dil = cv2.dilate(closed, dil_kernel, iterations=1)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w*h < max(300, int(200*scale)):
                continue
            boxes.append([x,y,w,h])
        boxes_sorted = sorted(boxes, key=lambda b:(b[0], b[1]))
        merged = []
        for b in boxes_sorted:
            if not merged:
                merged.append(b)
            else:
                last = merged[-1]
                if b[0] <= last[0]+last[2]+int(40*scale) and b[1] <= last[1]+last[3]+int(40*scale):
                    nx = min(last[0], b[0]); ny = min(last[1], b[1])
                    nw = max(last[0]+last[2], b[0]+b[2]) - nx
                    nh = max(last[1]+last[3], b[1]+b[3]) - ny
                    merged[-1] = [nx, ny, nw, nh]
                else:
                    merged.append(b)
        return [{"x":int(b[0]), "y":int(b[1]), "w":int(b[2]), "h":int(b[3])} for b in merged]
    except Exception as e:
        log(f"detect_boards_from_mask error: {e}")
        return []

def analyze_image_bytes(img_bytes):
    if np is None or cv2 is None or Image is None:
        log("Image libs missing.")
        return {"boards": [], "summary": {"total_blobs":0}}
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h,w = np_img.shape[:2]
    scale = 1.0
    if max(h,w) > 1600:
        scale = 1600.0 / max(h,w)
        np_img = cv2.resize(np_img, (int(w*scale), int(h*scale)))
        h,w = np_img.shape[:2]

    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    mask_b = cv2.inRange(hsv, (90,60,60), (140,255,255))
    mask_r1 = cv2.inRange(hsv, (0,60,60), (10,255,255))
    mask_r2 = cv2.inRange(hsv, (170,60,60), (180,255,255))
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask_any = cv2.bitwise_or(mask_b, mask_r)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_any>0).astype("uint8")*255)
    blobs = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_SIZE: continue
        cx = int(centroids[i][0]); cy = int(centroids[i][1])
        color = 'P' if mask_b[cy,cx] > 0 else ('B' if mask_r[cy,cx] > 0 else 'U')
        blobs.append({"x":cx, "y":cy, "area":area, "color":color})

    if len(blobs) == 0:
        return {"boards": [], "summary": {"total_blobs":0}}

    # detect board boxes
    boxes = detect_boards_from_mask(mask_any, scale)
    boards = []
    long_count = 0; super_long_count = 0; longish_count = 0

    if not boxes:
        # fallback to DBSCAN clustering
        pts = np.array([[b["x"], b["y"]] for b in blobs])
        labels_db = None
        try:
            if SKLEARN_AVAILABLE and len(pts)>0:
                db = DBSCAN(eps=max(30, int(80*scale)), min_samples=1).fit(pts)
                labels_db = db.labels_
            else:
                labels_db = np.zeros(len(pts), dtype=int)
        except Exception as e:
            log(f"DBSCAN fallback error: {e}")
            labels_db = np.zeros(len(pts), dtype=int)
        clusters={}
        for i,lb in enumerate(labels_db):
            clusters.setdefault(int(lb), []).append(blobs[i])
        for cid, items in clusters.items():
            items_sorted = sorted(items, key=lambda b:(b["x"], b["y"]))
            cols=[]; seq=[]
            for it in items_sorted:
                if not cols:
                    cols.append([it])
                else:
                    last = cols[-1][-1]
                    if abs(it["x"] - last["x"]) <= max(20, int(24*scale)):
                        cols[-1].append(it)
                    else:
                        cols.append([it])
            for col in cols:
                col_sorted = sorted(col, key=lambda b:b["y"])
                seq.extend([p["color"] for p in col_sorted])
            runs=[]; max_run=0
            if seq:
                cur=seq[0]; ln=1
                for s in seq[1:]:
                    if s==cur: ln+=1
                    else: runs.append((cur,ln)); cur=s; ln=1
                runs.append((cur,ln)); max_run = max(r[1] for r in runs)
            cat='other'
            if max_run>=10:
                cat='super_long'; super_long_count+=1; long_count+=1
            elif max_run>=8:
                cat='long'; long_count+=1
            elif max_run>=4:
                cat='longish'; longish_count+=1
            boards.append({"cluster":cid,"count":len(items),"max_run":max_run,"category":cat,"runs":runs})
    else:
        for i,box in enumerate(boxes):
            x,y,wid,hei = box["x"], box["y"], box["w"], box["h"]
            items = [b for b in blobs if (b["x"] >= x and b["x"] <= x+wid and b["y"] >= y and b["y"] <= y+hei)]
            items_sorted = sorted(items, key=lambda b:(b["x"], b["y"]))
            cols=[]; seq=[]
            for it in items_sorted:
                if not cols:
                    cols.append([it])
                else:
                    last = cols[-1][-1]
                    if abs(it["x"] - last["x"]) <= max(20, int(24*scale)):
                        cols[-1].append(it)
                    else:
                        cols.append([it])
            for col in cols:
                col_sorted = sorted(col, key=lambda b:b["y"])
                seq.extend([p["color"] for p in col_sorted])
            runs=[]; max_run=0
            if seq:
                cur=seq[0]; ln=1
                for s in seq[1:]:
                    if s==cur: ln+=1
                    else: runs.append((cur,ln)); cur=s; ln=1
                runs.append((cur,ln)); max_run = max(r[1] for r in runs)
            cat='other'
            if max_run>=10:
                cat='super_long'; super_long_count+=1; long_count+=1
            elif max_run>=8:
                cat='long'; long_count+=1
            elif max_run>=4:
                cat='longish'; longish_count+=1
            boards.append({"cluster": i, "box": box, "count": len(items), "max_run": max_run, "category": cat, "runs": runs})

    summary = {"total_blobs": len(blobs), "board_clusters": len(boards),
               "long_count": long_count, "super_long_count": super_long_count, "longish_count": longish_count}
    return {"boards": boards, "summary": summary}

# ---------- Page entry attempt ----------
def attempt_enter(page):
    try_texts = ["Free", "Free Play", "免费试玩", "免费", "试玩"]
    for t in try_texts:
        try:
            el = page.query_selector(f"text={t}")
            if el:
                try:
                    el.click(timeout=3000)
                    page.wait_for_timeout(1500)
                    log(f"Clicked text '{t}'")
                    return True
                except: pass
        except: pass
    try:
        btn = page.query_selector("button")
        if btn:
            try:
                btn.click(); page.wait_for_timeout(1200); log("Clicked generic button"); return True
            except: pass
    except: pass
    try:
        el = page.query_selector("[role=slider], .slider, .drag, .verify-slider, .slide-btn")
        if el:
            box = el.bounding_box()
            if box:
                sx = box["x"] + 2; sy = box["y"] + box["height"]/2
                ex = box["x"] + box["width"] - 4
                page.mouse.move(sx, sy); page.mouse.down(); page.mouse.move(ex, sy, steps=20); page.mouse.up()
                page.wait_for_timeout(1200)
                log("Attempted slider drag")
                return True
    except Exception as e:
        log(f"Slider attempt error: {e}")
    return False

def capture_aggregate(page, samples=CAPTURE_SAMPLES, delay_between=0.8):
    analyses = []
    last_img = None
    for i in range(samples):
        try:
            img_bytes = page.screenshot(full_page=True)
            last_img = img_bytes
            analysis = analyze_image_bytes(img_bytes)
            analyses.append(analysis)
            time.sleep(delay_between)
        except Exception as e:
            log(f"capture error: {e}")
            time.sleep(delay_between)
    if not analyses:
        return None, last_img
    max_long = max(a.get("summary", {}).get("long_count",0) for a in analyses)
    max_super = max(a.get("summary", {}).get("super_long_count",0) for a in analyses)
    max_longish = max(a.get("summary", {}).get("longish_count",0) for a in analyses)
    max_clusters = max(a.get("summary", {}).get("board_clusters",0) for a in analyses)
    total_blobs = max(a.get("summary", {}).get("total_blobs",0) for a in analyses)
    best = max(analyses, key=lambda x: x.get("summary", {}).get("board_clusters", 0))
    aggregated = {"summary": {"long_count": max_long, "super_long_count": max_super, "longish_count": max_longish, "board_clusters": max_clusters, "total_blobs": total_blobs}, "boards": best.get("boards", [])}
    return aggregated, last_img

# ---------- Decision logic (方案 A：把 longish 纳入中等判定/假信号) ----------
def decide_overall(aggregated):
    if not aggregated or "summary" not in aggregated:
        return "no_data", aggregated
    s = aggregated["summary"]
    long_count = int(s.get("long_count", 0))
    super_long = int(s.get("super_long_count", 0))
    longish = int(s.get("longish_count", 0))
    clusters = int(s.get("board_clusters", 0))
    total_blobs = int(s.get("total_blobs", 0))

    # 假信号规则：把 longish 也计入弱支持；若 long+longish < 2 则为假信号，不提醒
    if (long_count + longish) < 2:
        sparse = sum(1 for b in aggregated.get("boards", []) if b.get("count", 0) < 6)
        if clusters > 0 and sparse >= clusters * 0.6:
            return "胜率调低（平台收割时段）", s
        else:
            return "胜率中等（平台收割中等时段）", s

    # 放水判定（严格）：至少 MIN_BOARDS_FOR_POW 张长龙/超长龙
    if long_count >= MIN_BOARDS_FOR_POW:
        return "放水时段（提高胜率）", s

    # 中等胜率（中上）：若 long_count >= MID_LONG_REQ OR long_count + longish >= MID_LONG_REQ
    if (long_count >= MID_LONG_REQ) or ((long_count + longish) >= MID_LONG_REQ):
        return "中等胜率（中上）", s

    sparse = sum(1 for b in aggregated.get("boards", []) if b.get("count", 0) < 6)
    if clusters > 0 and sparse >= clusters * 0.6:
        return "胜率调低（平台收割时段）", s

    return "胜率中等（平台收割中等时段）", s

# ---------- Estimate helpers (history-based median) ----------
def median(lst):
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    mid = n//2
    return s[mid] if n%2==1 else (s[mid-1]+s[mid])//2

# ---------- Main run ----------
def run_once():
    st = github_get_state(); sha = st.get("sha"); state = st.get("data") or {}
    active = state.get("active", False); active_since = state.get("active_since"); cooldown_until = int(state.get("cooldown_until", 0) or 0)
    history = state.get("history", [])  # list of past durations in minutes
    now_ms = int(time.time()*1000)
    log(f"Start run. prev_active={active}, active_since={active_since}, cooldown_until={cooldown_until}")

    if sync_playwright is None:
        log("Playwright missing; abort.")
        return

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context(viewport={"width":1280,"height":800})
            page = context.new_page()
        except Exception as e:
            log(f"Playwright launch error: {e}"); return

        aggregated=None; last_img=None
        for url in DG_URLS:
            try:
                log(f"Navigating to {url}")
                page.goto(url, timeout=30000); page.wait_for_timeout(1500)
                attempt_enter(page)
                aggregated, last_img = capture_aggregate(page, samples=CAPTURE_SAMPLES, delay_between=0.8)
                if aggregated and aggregated.get("summary",{}).get("total_blobs",0) > 0:
                    log(f"Captured & found blobs on {url}")
                    break
                else:
                    log(f"No blobs found on {url} after attempt; trying next if any.")
            except Exception as e:
                log(f"Navigation error {url}: {e}")
                continue
        try: browser.close()
        except: pass

    if not aggregated or aggregated.get("summary",{}).get("total_blobs",0) == 0:
        log("No valid detection data (no blobs). Will not change state nor notify.")
        state.update({"active": active, "active_since": active_since, "last_seen": format_dt(now_dt()), "history": history})
        github_put_state(state, sha=sha, message="no data seen")
        return

    overall, summary = decide_overall(aggregated)
    log(f"Decision => {overall}  summary={summary}")
    for b in aggregated.get("boards", []):
        log(f"Board: max_run={b.get('max_run')} category={b.get('category')} count={b.get('count')}")

    now_iso_str = format_dt(now_dt())
    in_cooldown = now_ms < cooldown_until

    # If detected start condition
    if overall in ("放水时段（提高胜率）", "中等胜率（中上）"):
        if in_cooldown:
            log("In cooldown; skipping notify.")
            state.update({"active": active, "active_since": active_since, "last_seen": now_iso_str, "history": history, "cooldown_until": cooldown_until})
            github_put_state(state, sha=sha, message="cooldown skip")
            return

        if active:
            # already active -> update last_seen
            state.update({"active": True, "active_since": active_since, "last_seen": now_iso_str, "history": history, "cooldown_until": 0})
            github_put_state(state, sha=sha, message="active seen")
            log("Already active; no new start notify.")
            return
        else:
            # new activation -> estimate duration using median(history) or default
            estimated = median(history) or ESTIMATED_DEFAULT_MINUTES
            est_minutes = int(estimated)
            start_dt = now_dt()
            est_end_dt = start_dt + timedelta(minutes=est_minutes)
            # build message (human-readable time format)
            text = (
                f"💵📈[DG提醒] 发现放水/中上局势\n"
                f"判定: {overall}\n"
                f"长龙(>=8) 桌数: {int(summary.get('long_count',0))}\n"
                f"超长龙(>=10): {int(summary.get('super_long_count',0))}\n"
                f"检测时间: {format_dt(start_dt)}\n"
                f"⚠️预计持续时长（估计）: {est_minutes} 分钟\n"
                f"⚠️预计结束时间（估计）: {format_dt(est_end_dt)}\n"
                f"⚠️局势预计：剩下 {est_minutes} 分鐘（基於歷史中位數估計）\n"
                "說明: 按既定閾值觸發；此為估計值，結束時會重新回報真實持續時間。"
            )
            send_telegram(text)
            # save state active
            state = {"active": True, "active_since": start_dt.isoformat(), "last_seen": format_dt(start_dt), "history": history, "cooldown_until": 0}
            github_put_state(state, sha=sha, message="start active")
            log("Sent start notification with estimate.")
            return
    else:
        # Non-remind categories
        if active:
            # previously active -> ended
            try:
                start_dt = datetime.fromisoformat(active_since)
                end_dt = now_dt()
                dur_min = minutes_between(start_dt, end_dt)
                # append to history
                history2 = history or []
                history2.append(dur_min)
                # keep limited history size
                if len(history2) > 50:
                    history2 = history2[-50:]
                text = (
                    f"⚠️[DG提醒] 放水已結束\n"
                    f"開始: {format_dt(start_dt)}\n"
                    f"結束: {format_dt(end_dt)}\n"
                    f"共持續 (實際): {dur_min} 分鐘\n"
                    f"已將本次持續時長加入歷史，用於未來估計。"
                )
                send_telegram(text)
                log("Sent end notification.")
                # set cooldown
                cd_ms = COOLDOWN_MINUTES * 60 * 1000
                new_state = {"active": False, "active_since": None, "last_seen": format_dt(end_dt), "cooldown_until": int(time.time()*1000) + cd_ms, "history": history2}
                github_put_state(new_state, sha=sha, message="end active")
            except Exception as e:
                log(f"Error computing duration or saving history: {e}")
                # still try to save state without history change
                new_state = {"active": False, "active_since": None, "last_seen": format_dt(now_dt()), "cooldown_until": int(time.time()*1000) + COOLDOWN_MINUTES*60*1000, "history": history}
                github_put_state(new_state, sha=sha, message="end active error")
            return
        else:
            # nothing active and nothing to notify
            state.update({"active": False, "active_since": None, "last_seen": now_iso_str, "cooldown_until": state.get("cooldown_until", 0), "history": history})
            github_put_state(state, sha=sha, message="no change")
            log("No active condition; nothing to notify.")
            return

def main():
    try:
        run_once()
    except Exception as e:
        log(f"Unhandled exception: {e}")
        traceback.print_exc()
        try:
            st = github_get_state(); sha = st.get("sha"); sdat = st.get("data") or {}; sdat.update({"last_error": str(e), "last_error_time": format_dt(now_dt())}); github_put_state(sdat, sha=sha, message="error state")
        except Exception as ex:
            log(f"Failed saving error state: {ex}")
    finally:
        try: sys.exit(0)
        except SystemExit: pass

if __name__ == "__main__":
    main()

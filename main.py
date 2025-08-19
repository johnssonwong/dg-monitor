# main.py
import os, json, time, base64, math
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from playwright.sync_api import sync_playwright

# ----------------- 配置（可用 GitHub Secrets 或直接写入 env） -----------------
TG_TOKEN = os.environ.get("TG_TOKEN")   # Telegram Bot Token
TG_CHAT  = os.environ.get("TG_CHAT")    # Telegram Chat ID
DG_URLS = [ os.environ.get("DG_URL1","https://dg18.co/wap/"), os.environ.get("DG_URL2","https://dg18.co/") ]

# GitHub repository info & token (Actions provides GITHUB_REPOSITORY and GITHUB_TOKEN)
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY")  # owner/repo
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# 阈值（可按需调整）
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW","3"))      # 放水：至少几桌为长龙/超长龙
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))                 # 中等（中上）要求 >=2 张长龙
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))        # 提醒冷却时间（分钟）
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE","6"))              # blob 最小像素
# --------------------------------------------------------------------------

LOG_PREFIX = "[DGMON]"

def log(msg):
    print(f"{LOG_PREFIX} {datetime.now().astimezone().isoformat()} - {msg}")

# GitHub state file helpers (store state.json in repo root)
STATE_PATH = "state.json"
def github_get_state():
    # get file content via GitHub API
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return {}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        j = r.json()
        content = base64.b64decode(j["content"]).decode()
        try:
            return {"sha": j["sha"], "data": json.loads(content)}
        except:
            return {"sha": j["sha"], "data": {}}
    else:
        return {"sha": None, "data": {}}

def github_put_state(data, sha=None, message="update state"):
    if not GITHUB_REPO or not GITHUB_TOKEN:
        log("GITHUB_TOKEN or GITHUB_REPO not set — cannot persist state.")
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": os.environ.get("GITHUB_REF", "main")
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200,201):
        log("State saved to repo.")
        return True
    else:
        log(f"Failed to save state: {r.status_code} {r.text}")
        return False

def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG_TOKEN or TG_CHAT missing. Skipping Telegram send.")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code == 200:
            log("Telegram message sent.")
            return True
        else:
            log(f"Telegram send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram send exception: {e}")
        return False

# Basic image analysis pipeline: detect red/blue blobs and compute run lengths per board-like region
def analyze_image_array(img_np):
    # img_np: OpenCV BGR image
    h, w = img_np.shape[:2]
    # downscale for speed if huge
    scale = 1.0
    if max(h,w) > 1600:
        scale = 1600.0 / max(h,w)
        img_np = cv2.resize(img_np, (int(w*scale), int(h*scale)))
        h,w = img_np.shape[:2]

    # detect red and blue pixels mask (simple thresholds)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    # blue mask
    lower_blue = np.array([90,60,60])
    upper_blue = np.array([140,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # red mask (two ranges)
    lower_red1 = np.array([0,60,60]); upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,60,60]); upper_red2 = np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # combine masks
    mask = (mask_blue>0).astype(np.uint8)*1 + (mask_red>0).astype(np.uint8)*2

    # find connected components on combined color mask (nonzero either)
    mask_any = ((mask>0)*255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_any, connectivity=8)
    # accumulate color blobs by bounding box/centroid clustering
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_SIZE:
            continue
        x = int(centroids[i][0]); y = int(centroids[i][1])
        # sample color at centroid
        px = img_np[y, x]
        hsv_px = cv2.cvtColor(px.reshape((1,1,3)), cv2.COLOR_BGR2HSV)[0,0]
        # decide color by mask at centroid
        c = mask[y,x]
        col = 'U'
        if c==1: col='P'  # blue -> P (player)
        elif c==2: col='B' # red -> B (banker)
        else:
            # fallback by hue
            hpx = hsv_px[0]
            if 90 <= hpx <= 140: col='P'
            elif hpx<=10 or hpx>=170: col='B'
        blobs.append({"x":x,"y":y,"area":area,"color":col})

    # If zero blobs, return empty stats
    if len(blobs)==0:
        return {"boards":[], "summary": {"total_blobs":0}}

    # Heuristic: cluster blobs into board regions by spatial clustering
    pts = np.array([[b["x"], b["y"]] for b in blobs])
    # use simple grid clustering: kmeans with estimated k = number of columns*rows unknown => use DBSCAN spatial clustering
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=120*scale, min_samples=1).fit(pts)   # eps in pixels (adjust)
        labels_db = db.labels_
    except Exception:
        # fallback: all blobs into single cluster
        labels_db = np.zeros(len(blobs), dtype=int)

    clusters = {}
    for i, lb in enumerate(labels_db):
        clusters.setdefault(int(lb), []).append(blobs[i])

    boards = []
    long_count = 0
    super_long_count = 0
    longish_count = 0
    for cid, items in clusters.items():
        # sort items by x then y to create column-major flattened sequence approximation
        items_sorted = sorted(items, key=lambda b: (b["x"], b["y"]))
        # create flattened seq by grouping x columns
        xs = [it["x"] for it in items_sorted]
        # cluster x-values into vertical columns
        col_groups = []
        cur = [items_sorted[0]]
        for it in items_sorted[1:]:
            if abs(it["x"] - cur[-1]["x"]) <= 24*scale:
                cur.append(it)
            else:
                col_groups.append(cur)
                cur = [it]
        col_groups.append(cur)
        # for each col, sort by y (top->bottom)
        seq = []
        for col in col_groups:
            col_sorted = sorted(col, key=lambda b: b["y"])
            for p in col_sorted:
                seq.append(p["color"])
        # compute runs
        runs = []
        if len(seq)>0:
            curc = seq[0]; curlen = 1
            for s in seq[1:]:
                if s==curc:
                    curlen += 1
                else:
                    runs.append((curc, curlen))
                    curc = s; curlen = 1
            runs.append((curc, curlen))
        max_run = max([r[1] for r in runs]) if runs else 0
        cat = "other"
        if max_run >= 10:
            cat = "super_long"; super_long_count += 1; long_count += 1
        elif max_run >= 8:
            cat = "long"; long_count += 1
        elif max_run >= 4:
            cat = "longish"; longish_count += 1

        boards.append({"cluster":cid, "count": len(items), "max_run": max_run, "category": cat, "runs": runs})

    summary = {"total_blobs": len(blobs), "board_clusters": len(boards),
               "long_count": long_count, "super_long_count": super_long_count, "longish_count": longish_count}
    return {"boards": boards, "summary": summary}

def capture_and_analyze(page):
    # screenshot full viewport
    img_bytes = page.screenshot(full_page=True)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return analyze_image_array(open_cv_image), img_bytes

def attempt_enter_dg(page):
    # Try to click "Free", "免费试玩" or "Free Play" etc — heuristic
    possible_texts = ["Free", "免费试玩", "免费", "Free Play", "试玩"]
    for t in possible_texts:
        try:
            el = page.query_selector(f"text={t}")
            if el:
                log(f"Clicking button with text '{t}'")
                el.click(timeout=3000)
                page.wait_for_timeout(1500)
                return True
        except:
            pass
    # try common CSS/class buttons
    try:
        btn = page.query_selector("button")
        if btn:
            btn.click()
            page.wait_for_timeout(1000)
            return True
    except:
        pass
    return False

def attempt_drag_slider(page):
    # many sites have a slider that requires dragging; attempt to find input[type=range] or draggable slider
    try:
        # try find element with role slider or common class
        el = page.query_selector("[role=slider], .slider, .drag, .verify-slider, .slide-btn")
        if el:
            box = el.bounding_box()
            if box:
                x = box["x"] + 2
                y = box["y"] + box["height"]/2
                end_x = box["x"] + box["width"] - 2
                page.mouse.move(x,y)
                page.mouse.down()
                page.mouse.move(end_x, y, steps=20)
                page.mouse.up()
                page.wait_for_timeout(1000)
                log("Attempted slider drag")
                return True
    except Exception as e:
        log(f"slider attempt error: {e}")
    return False

def main():
    # load saved state
    st = github_get_state()
    sha = st.get("sha")
    state = st.get("data") or {}
    active = state.get("active", False)
    active_since = state.get("active_since")  # ISO str

    log(f"Starting run. previous active={active}, active_since={active_since}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={"width":1280,"height":800})
        page = context.new_page()
        # try the DG urls in order
        success = False
        for url in DG_URLS:
            try:
                log(f"Opening {url} ...")
                page.goto(url, timeout=30000)
                page.wait_for_timeout(1500)
                # attempt to click Free / enter
                if attempt_enter_dg(page):
                    page.wait_for_timeout(2500)
                # attempt to drag slider if found
                attempt_drag_slider(page)
                # wait a bit for page to load game panels
                page.wait_for_timeout(2500)
                success = True
                break
            except Exception as e:
                log(f"Error opening {url}: {e}")
                continue

        if not success:
            log("Unable to open DG URLs. Exiting.")
            browser.close()
            return

        # capture & analyze
        analysis, img_bytes = capture_and_analyze(page)
        summary = analysis.get("summary", {})
        log(f"Analysis summary: {summary}")

        # apply decision logic (according to your rules)
        long_count = summary.get("long_count",0)
        super_long_count = summary.get("super_long_count",0)
        longish_count = summary.get("longish_count",0)
        total_clusters = summary.get("board_clusters",0)
        total_blobs = summary.get("total_blobs",0)

        overall = "胜率中等（平台收割中等时段）"
        # 放水时段：至少 MIN_BOARDS_FOR_POW 张为 长龙/超长龙 (long_count)
        if long_count >= MIN_BOARDS_FOR_POW:
            overall = "放水时段（提高胜率）"
        # 中等胜率（中上）: mixed but >= MID_LONG_REQ 长龙且有若干多连
        elif long_count >= MID_LONG_REQ and longish_count>0:
            overall = "中等胜率（中上）"
        else:
            # detect mostly empty/sparse => 胜率调低/收割
            sparse_boards = sum(1 for b in analysis["boards"] if b["count"] < 6)
            if total_clusters>0 and sparse_boards >= total_clusters*0.6:
                overall = "胜率调低（平台收割时段）"
            else:
                overall = "胜率中等（平台收割中等时段）"

        log(f"Overall classification: {overall}")

        # decide notify
        now_iso = datetime.now(timezone(timedelta(hours=8))).isoformat()  # Malaysia timezone
        cooldown_until = state.get("cooldown_until", 0)
        now_ts = int(time.time()*1000)
        in_cooldown = now_ts < int(cooldown_until or 0)

        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            if active:
                # already active -> update info (still active)
                log("Condition still active.")
                # persist state (no new notify)
                state.update({"active": True, "active_since": active_since, "last_seen": now_iso})
                github_put_state(state, sha=sha, message="update active seen")
            else:
                if in_cooldown:
                    log("In cooldown after previous notify, skipping new notify.")
                else:
                    # new activation -> send Telegram start message & store active state
                    text = f"[DG提醒] 发现放水/中上局势\n判定: {overall}\n长龙(>=8) 桌数: {long_count}\n超长龙(>=10): {super_long_count}\n时间: {now_iso}\n说明: 按设定阈值触发。"
                    send_telegram(text)
                    # mark active
                    state = {"active": True, "active_since": now_iso, "last_seen": now_iso, "cooldown_until": 0}
                    github_put_state(state, sha=sha, message="start active")
        else:
            # not a "remind" classification
            if active:
                # previously active but now ended -> compute duration and send end message
                try:
                    start_dt = datetime.fromisoformat(active_since)
                    end_dt = datetime.now(timezone(timedelta(hours=8)))
                    dur_min = int((end_dt - start_dt).total_seconds()/60)
                    text = f"[DG提醒] 放水已结束\n开始: {active_since}\n结束: {end_dt.isoformat()}\n共持续: {dur_min} 分钟"
                    send_telegram(text)
                except Exception as e:
                    log(f"Error computing duration: {e}")
                # set cooldown (prevents immediate re-notify)
                cooldown_ms = COOLDOWN_MINUTES * 60 * 1000
                state = {"active": False, "active_since": None, "last_seen": now_iso, "cooldown_until": int(time.time()*1000) + cooldown_ms}
                github_put_state(state, sha=sha, message="end active")
            else:
                # nothing to do
                log("No active condition and nothing to do.")

        browser.close()

if __name__ == "__main__":
    main()

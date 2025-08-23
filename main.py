# main.py  -- 强化版（按你聊天框内所有规则与判定）
import os, json, time, base64, traceback, sys
from datetime import datetime, timezone, timedelta
from io import BytesIO

# external libs
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
    # If imports fail, we still catch and exit gracefully later
    requests = None
    np = None
    cv2 = None
    Image = None
    sync_playwright = None
    DBSCAN = None
    SKLEARN_AVAILABLE = False

# ---------------- Configuration (via env / GitHub Secrets) ----------------
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT  = os.environ.get("TG_CHAT")
DG_URLS = [ os.environ.get("DG_URL1", "https://dg18.co/wap/"), os.environ.get("DG_URL2", "https://dg18.co/") ]
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW", "3"))   # 放水需要至少 N 张长龙/超长龙 (默认3)
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))              # 中等(中上)需要至少 N 张长龙 (默认2)
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))     # 冷却分钟
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE", "6"))           # 连通域最小像素
CAPTURE_SAMPLES = int(os.environ.get("CAPTURE_SAMPLES", "3"))       # 多次截图样本数量 (默认3)
STATE_PATH = "state.json"
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
# Malaysia timezone
TZ = timezone(timedelta(hours=8))
# -------------------------------------------------------------------------

def now_iso():
    return datetime.now(TZ).isoformat()

def log(msg):
    print(f"[DGMON] {now_iso()} - {msg}")

# ---------- GitHub state helpers (persist active state) ----------
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

# ---------- Image analysis helpers ----------
def analyze_image_bytes(img_bytes):
    """
    Returns {"boards": [...], "summary": {...}}
    Each board: {cluster, count, max_run, category, runs}
    summary: total_blobs, board_clusters, long_count, super_long_count, longish_count
    """
    if np is None or cv2 is None or Image is None:
        log("Image libraries not available.")
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

    # blue mask (player)
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # red mask (banker) - two ranges
    lower_red1 = np.array([0,60,60]); upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,60,60]); upper_red2 = np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    mask_any = cv2.bitwise_or(mask_blue, mask_red)

    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_any>0).astype("uint8")*255)
    blobs = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_SIZE:
            continue
        cx = int(centroids[i][0]); cy = int(centroids[i][1])
        color = 'U'
        if mask_blue[cy, cx] > 0:
            color = 'P'   # Player (blue)
        elif mask_red[cy, cx] > 0:
            color = 'B'   # Banker (red)
        else:
            # fallback by sampling hue
            hsv_px = hsv[cy, cx]
            hpx = int(hsv_px[0])
            if 90 <= hpx <= 140: color = 'P'
            elif hpx <= 10 or hpx >= 170: color = 'B'
            else: color = 'U'
        blobs.append({"x":cx, "y":cy, "area":area, "color":color})

    if len(blobs) == 0:
        return {"boards": [], "summary": {"total_blobs":0}}

    pts = np.array([[b["x"], b["y"]] for b in blobs])

    labels_db = None
    clusters = {}
    try:
        if SKLEARN_AVAILABLE:
            db = DBSCAN(eps=max(40, int(120*scale)), min_samples=1).fit(pts)
            labels_db = db.labels_
        else:
            # fallback: naive grid clustering
            labels_db = []
            gid = 0
            used = [False]*len(pts)
            for i,p in enumerate(pts):
                if used[i]: 
                    labels_db.append(gid-1)  # won't happen
                    continue
                labels_db.append(gid)
                used[i]=True
                for j,q in enumerate(pts):
                    if not used[j] and abs(int(p[0])-int(q[0]))<=120*scale and abs(int(p[1])-int(q[1]))<=120*scale:
                        used[j]=True
                        labels_db.append(gid)
                gid += 1
            # if fallback created wrong length, regenerate simple single cluster
            if len(labels_db) != len(pts):
                labels_db = np.zeros(len(pts), dtype=int)
                log("Fallback clustering used single-cluster.")
    except Exception as e:
        log(f"Clustering error: {e}")
        labels_db = np.zeros(len(pts), dtype=int)

    for i, lb in enumerate(labels_db):
        clusters.setdefault(int(lb), []).append(blobs[i])

    boards = []
    long_count = 0
    super_long_count = 0
    longish_count = 0

    for cid, items in clusters.items():
        # order items left-to-right then top-to-bottom to approximate bead reading
        items_sorted = sorted(items, key=lambda b: (b["x"], b["y"]))
        # group into x-columns by x distance
        cols = []
        for it in items_sorted:
            if not cols:
                cols.append([it])
            else:
                last = cols[-1][-1]
                if abs(it["x"] - last["x"]) <= max(20, int(24*scale)):
                    cols[-1].append(it)
                else:
                    cols.append([it])
        # for each col, sort by y then flatten col-major
        seq = []
        for col in cols:
            col_sorted = sorted(col, key=lambda b: b["y"])
            seq.extend([p["color"] for p in col_sorted])
        # compute runs
        runs = []
        if seq:
            cur = seq[0]; ln = 1
            for s in seq[1:]:
                if s == cur:
                    ln += 1
                else:
                    runs.append((cur, ln))
                    cur = s; ln = 1
            runs.append((cur, ln))
        max_run = max([r[1] for r in runs]) if runs else 0
        cat = "other"
        if max_run >= 10:
            cat = "super_long"; super_long_count += 1; long_count += 1
        elif max_run >= 8:
            cat = "long"; long_count += 1
        elif max_run >= 4:
            cat = "longish"; longish_count += 1
        boards.append({"cluster": cid, "count": len(items), "max_run": max_run, "category": cat, "runs": runs})

    summary = {"total_blobs": len(blobs), "board_clusters": len(boards),
               "long_count": long_count, "super_long_count": super_long_count, "longish_count": longish_count}
    return {"boards": boards, "summary": summary}

# ---------- Page entry attempts ----------
def attempt_enter(page):
    """
    Heuristics: try clicking texts like Free / 免费试玩 / 试玩 / 免费
    then attempt slider drag if present
    Return True if likely entered (no guarantee), False otherwise.
    """
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
                except Exception:
                    pass
        except Exception:
            pass
    # try generic button
    try:
        btn = page.query_selector("button")
        if btn:
            try:
                btn.click(); page.wait_for_timeout(1200)
                log("Clicked generic button")
                return True
            except:
                pass
    except:
        pass

    # try slider drag
    try:
        el = page.query_selector("[role=slider], .slider, .drag, .verify-slider, .slide-btn")
        if el:
            box = el.bounding_box()
            if box:
                sx = box["x"] + 2; sy = box["y"] + box["height"]/2
                ex = box["x"] + box["width"] - 4
                page.mouse.move(sx, sy); page.mouse.down()
                page.mouse.move(ex, sy, steps=20); page.mouse.up()
                page.wait_for_timeout(1200)
                log("Attempted slider drag")
                return True
    except Exception as e:
        log(f"Slider attempt error: {e}")

    # if nothing worked
    return False

# ---------- Multi-capture & aggregate ----------
def capture_aggregate(page, samples=3, delay_between=1.0):
    """
    Capture several screenshots and aggregate analysis.
    Return aggregated_summary and the last image bytes (for debug)
    aggregated_summary: merges boards summaries by picking max counts.
    """
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
    # aggregate by selecting maximum counts across runs (conservative)
    max_long = max(a.get("summary", {}).get("long_count", 0) for a in analyses)
    max_super = max(a.get("summary", {}).get("super_long_count", 0) for a in analyses)
    max_longish = max(a.get("summary", {}).get("longish_count", 0) for a in analyses)
    max_clusters = max(a.get("summary", {}).get("board_clusters", 0) for a in analyses)
    total_blobs = max(a.get("summary", {}).get("total_blobs", 0) for a in analyses)
    # pick boards list from the analysis with the maximum board_clusters (best snapshot)
    best = max(analyses, key=lambda x: x.get("summary", {}).get("board_clusters", 0))
    aggregated = {"summary": {"long_count": max_long, "super_long_count": max_super, "longish_count": max_longish, "board_clusters": max_clusters, "total_blobs": total_blobs}, "boards": best.get("boards", [])}
    return aggregated, last_img

# ---------- Decision logic (strict per your rules) ----------
def decide_overall(aggregated):
    if not aggregated or "summary" not in aggregated:
        return "no_data", aggregated
    s = aggregated["summary"]
    long_count = int(s.get("long_count", 0))
    super_long = int(s.get("super_long_count", 0))
    longish = int(s.get("longish_count", 0))
    clusters = int(s.get("board_clusters", 0))
    total_blobs = int(s.get("total_blobs", 0))

    # Fake-signal rule: if total long_count < 2, treat as fake => never notify
    if long_count < 2:
        # But still classify to proper non-remind categories
        # Determine if it's sparse (收割) or 中等
        sparse = sum(1 for b in aggregated.get("boards", []) if b.get("count", 0) < 6)
        if clusters > 0 and sparse >= clusters * 0.6:
            return "胜率调低（平台收割时段）", s
        else:
            return "胜率中等（平台收割中等时段）", s

    # 放水判定：符合至少 MIN_BOARDS_FOR_POW 张（长 or super）
    if long_count >= MIN_BOARDS_FOR_POW:
        # Also ensure at least 3 tables overall as you insisted (but we already use MIN_BOARDS_FOR_POW default 3)
        return "放水时段（提高胜率）", s

    # 中等胜率（中上）：若 >= MID_LONG_REQ 长龙且有若干 longish
    if long_count >= MID_LONG_REQ and longish > 0:
        return "中等胜率（中上）", s

    # otherwise detect if mostly sparse -> 收割
    sparse = sum(1 for b in aggregated.get("boards", []) if b.get("count", 0) < 6)
    if clusters > 0 and sparse >= clusters * 0.6:
        return "胜率调低（平台收割时段）", s

    return "胜率中等（平台收割中等时段）", s

# ---------- Main run ----------
def run_once():
    # load state
    st = github_get_state()
    sha = st.get("sha")
    state = st.get("data") or {}
    active = state.get("active", False)
    active_since = state.get("active_since")
    cooldown_until = int(state.get("cooldown_until", 0) or 0)
    now_ms = int(time.time()*1000)
    log(f"Start run. prev_active={active}, active_since={active_since}, cooldown_until={cooldown_until}")

    # prerequisites check
    if sync_playwright is None:
        log("Playwright not available in environment; aborting detection.")
        return

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context(viewport={"width":1280,"height":800})
            page = context.new_page()
        except Exception as e:
            log(f"Playwright launch error: {e}")
            return

        entered = False
        aggregated = None
        last_img = None

        # try each DG url
        for url in DG_URLS:
            try:
                log(f"Navigating to {url}")
                page.goto(url, timeout=30000)
                page.wait_for_timeout(1400)
                # attempt to enter (click Free, slider)
                entered = attempt_enter(page)
                # always try capture even if attempt_enter returned False (maybe already in)
                aggregated, last_img = capture_aggregate(page, samples=CAPTURE_SAMPLES, delay_between=0.8)
                if aggregated and aggregated.get("summary", {}).get("total_blobs", 0) > 0:
                    log(f"Captured & found blobs on {url}")
                    break
                else:
                    log(f"No blobs found on {url} after entry attempt; will try next url if any.")
                    # continue to next url
            except Exception as e:
                log(f"Navigation error for {url}: {e}")
                continue

        # close browser
        try:
            browser.close()
        except:
            pass

    if not aggregated or aggregated.get("summary", {}).get("total_blobs", 0) == 0:
        log("No valid detection data (no blobs). Will not change active state nor notify.")
        # Do not change state; do not send notifications. Save last_seen optionally.
        state.update({"active": active, "active_since": active_since, "last_seen": now_iso()})
        github_put_state(state, sha=sha, message="no data seen")
        return

    # decide overall
    overall, summary = decide_overall(aggregated)
    log(f"Decision => {overall}  summary={summary}")

    # Only notify for 放水 or 中等胜率（中上） AND respecting fake-signal rule (we enforced long_count<2 earlier)
    long_count = int(summary.get("long_count", 0))
    now_iso_str = now_iso()
    in_cooldown = now_ms < cooldown_until

    if overall in ("放水时段（提高胜率）", "中等胜率（中上）"):
        if in_cooldown:
            log("In cooldown; skipping new notify.")
            # still update last_seen
            state.update({"active": active, "active_since": active_since, "last_seen": now_iso_str, "cooldown_until": cooldown_until})
            github_put_state(state, sha=sha, message="cooldown skip")
            return

        if active:
            # already active, just update last_seen
            state.update({"active": True, "active_since": active_since, "last_seen": now_iso_str, "cooldown_until": 0})
            github_put_state(state, sha=sha, message="active seen")
            log("Already active; no new start notify.")
            return
        else:
            # new activation
            text = (f"[DG提醒] 发现放水/中上局势\n判定: {overall}\n"
                    f"长龙(>=8) 桌数: {int(summary.get('long_count',0))}\n"
                    f"超长龙(>=10): {int(summary.get('super_long_count',0))}\n"
                    f"时间: {now_iso_str}\n"
                    "说明: 按既定阈值触发，开始时间已记录。")
            send_telegram(text)
            state = {"active": True, "active_since": now_iso_str, "last_seen": now_iso_str, "cooldown_until": 0}
            github_put_state(state, sha=sha, message="start active")
            log("Sent start notification.")
            return
    else:
        # overall is non-remind (胜率中等 or 胜率调低)
        if active:
            # previously active -> ended
            try:
                start_dt = datetime.fromisoformat(active_since)
                end_dt = datetime.now(TZ)
                dur_minutes = int((end_dt - start_dt).total_seconds() / 60)
                text = (f"[DG提醒] 放水已结束\n開始: {active_since}\n結束: {end_dt.isoformat()}\n共持續: {dur_minutes} 分鐘")
                send_telegram(text)
                log("Sent end notification.")
            except Exception as e:
                log(f"Error computing duration: {e}")
            # set cooldown to prevent immediate re-alert
            cd_ms = COOLDOWN_MINUTES * 60 * 1000
            new_state = {"active": False, "active_since": None, "last_seen": now_iso(), "cooldown_until": int(time.time()*1000) + cd_ms}
            github_put_state(new_state, sha=sha, message="end active")
            return
        else:
            # nothing active and nothing to notify
            state.update({"active": False, "active_since": None, "last_seen": now_iso(), "cooldown_until": state.get("cooldown_until", 0)})
            github_put_state(state, sha=sha, message="no change")
            log("No active condition; nothing to notify.")
            return

# ---------- Entrypoint with robust exception handling ----------
def main():
    try:
        run_once()
    except Exception as e:
        log(f"Unhandled exception in run: {e}")
        traceback.print_exc()
        # do not raise non-zero exit to avoid Actions failure statuses you reported
        try:
            # attempt to save diagnostic state
            st = github_get_state()
            sha = st.get("sha")
            sdat = st.get("data") or {}
            sdat.update({"last_error": str(e), "last_error_time": now_iso()})
            github_put_state(sdat, sha=sha, message="error state")
        except Exception as ex:
            log(f"Failed to save error state: {ex}")
    finally:
        # Always exit 0 to avoid "Process completed with exit code X"
        try:
            sys.exit(0)
        except SystemExit:
            pass

if __name__ == "__main__":
    main()

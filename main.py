# main.py (updated with improved detection, debug screenshot->Telegram, configurable params)
import os, json, time, base64
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from playwright.sync_api import sync_playwright

# ---------- ENV / config ----------
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT  = os.environ.get("TG_CHAT")
DEBUG    = os.environ.get("DEBUG","0")  # set to "1" to enable debug (screenshot + verbose)
DG_URLS = [ os.environ.get("DG_URL1","https://dg18.co/wap/"), os.environ.get("DG_URL2","https://dg18.co/") ]

# detection params (env override possible)
MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW","3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE","3"))  # lowered default

# color thresholds (HSV)
BLUE_H_LOW  = int(os.environ.get("BLUE_H_LOW","80"))
BLUE_H_HIGH = int(os.environ.get("BLUE_H_HIGH","160"))
SAT_MIN     = int(os.environ.get("SAT_MIN","50"))
VAL_MIN     = int(os.environ.get("VAL_MIN","50"))

# clustering params
DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS","60"))  # pixels (will be scaled by image resize)
COLUMN_GAP = float(os.environ.get("COLUMN_GAP","18"))

GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
STATE_PATH = "state.json"
TZ = timezone(timedelta(hours=8))

def log(msg):
    print(f"[DGMON] {datetime.now(TZ).isoformat()} - {msg}")

# ---------- GitHub state helpers ----------
def github_get_state():
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return {"sha": None, "data": {}}
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
        log("GitHub not configured, cannot save state.")
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": os.environ.get("GITHUB_REF","main")
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in (200,201):
        log("Saved state.json")
        return True
    else:
        log(f"Failed to save state: {r.status_code} {r.text}")
        return False

# ---------- Telegram helpers ----------
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG_TOKEN/TG_CHAT missing")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id":TG_CHAT,"text":text}, timeout=20)
        if r.status_code==200:
            log("Telegram message sent")
            return True
        else:
            log(f"Telegram send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram send exception: {e}")
        return False

def send_telegram_photo(img_bytes, caption=""):
    if not TG_TOKEN or not TG_CHAT:
        log("TG_TOKEN/TG_CHAT missing")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    try:
        files = {"photo": ("screenshot.jpg", img_bytes)}
        data = {"chat_id": TG_CHAT, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=30)
        if r.status_code==200:
            log("Telegram photo sent")
            return True
        else:
            log(f"Telegram photo send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram photo exception: {e}")
        return False

# ---------- image analysis ----------
def analyze_image_bytes(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h,w = np_img.shape[:2]
    scale = 1.0
    if max(h,w) > 1600:
        scale = 1600.0 / max(h,w)
        np_img = cv2.resize(np_img, (int(w*scale), int(h*scale)))
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([BLUE_H_LOW, SAT_MIN, VAL_MIN])
    upper_blue = np.array([BLUE_H_HIGH, 255, 255])
    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    # red
    mask_r1 = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (10,255,255))
    mask_r2 = cv2.inRange(hsv, (170, SAT_MIN, VAL_MIN), (180,255,255))
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask_any = cv2.bitwise_or(mask_b, mask_r)
    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_any>0).astype("uint8")*255)
    blobs=[]
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_SIZE: continue
        cx = int(centroids[i][0]); cy=int(centroids[i][1])
        color = 'P' if mask_b[cy,cx]>0 else ('B' if mask_r[cy,cx]>0 else 'U')
        blobs.append({"x":cx,"y":cy,"area":area,"color":color})
    summary = {"total_blobs": len(blobs)}
    if len(blobs)==0:
        return {"boards":[], "summary":summary, "scale": scale}
    # clustering (DBSCAN)
    try:
        from sklearn.cluster import DBSCAN
        pts = np.array([[b["x"], b["y"]] for b in blobs])
        eps = max(8, float(DBSCAN_EPS) * scale)  # scale eps too
        db = DBSCAN(eps=eps, min_samples=1).fit(pts)
        labels_db = db.labels_
    except Exception as e:
        log(f"DBSCAN error or missing sklearn: {e}; fallback to single cluster")
        labels_db = np.zeros(len(blobs), dtype=int)
    clusters={}
    for i,lb in enumerate(labels_db):
        clusters.setdefault(int(lb), []).append(blobs[i])
    boards=[]
    long_count=0; super_long=0; longish=0
    for cid, items in clusters.items():
        items_sorted = sorted(items, key=lambda b:(b["x"], b["y"]))
        # group into columns by x gap
        cols=[]
        for it in items_sorted:
            if not cols or abs(it["x"]-cols[-1][-1]["x"]) > max(6, COLUMN_GAP*scale):
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
            cur=seq[0]; ln=1
            for s in seq[1:]:
                if s==cur: ln+=1
                else:
                    runs.append((cur,ln)); cur=s; ln=1
            runs.append((cur,ln))
        max_run = max([r[1] for r in runs]) if runs else 0
        cat='other'
        if max_run>=10:
            cat='super_long'; super_long+=1; long_count+=1
        elif max_run>=8:
            cat='long'; long_count+=1
        elif max_run>=4:
            cat='longish'; longish+=1
        boards.append({"cluster":cid,"count":len(items),"max_run":max_run,"category":cat,"runs":runs,"flattened_seq": seq[:60]})
    summary.update({"board_clusters": len(boards),"long_count": long_count,"super_long_count": super_long,"longish_count": longish})
    return {"boards":boards,"summary":summary,"scale": scale}

# ---------- page helpers ----------
def attempt_enter_and_wait(page):
    texts = ["Free","Free Play","免费试玩","免费","试玩"]
    for t in texts:
        try:
            el = page.query_selector(f"text={t}")
            if el:
                el.click(timeout=3000)
                page.wait_for_timeout(1800)
                # after click wait for network idle
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except:
                    page.wait_for_timeout(1500)
                return True
        except:
            pass
    # try any button
    try:
        btn = page.query_selector("button")
        if btn:
            btn.click(); page.wait_for_timeout(1500)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except:
                page.wait_for_timeout(1500)
            return True
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
                page.wait_for_timeout(1500)
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except:
                    page.wait_for_timeout(1500)
                return True
    except: pass
    return False

def find_game_screenshot_bytes(page):
    # priority: canvas element screenshot or first visible iframe, else full page
    try:
        # try canvas
        canv = page.query_selector("canvas")
        if canv:
            try:
                b = canv.screenshot()
                log("Captured canvas element screenshot")
                return b
            except Exception as e:
                log(f"canvas screenshot failed: {e}")
        # try iframe with many nodes
        frames = page.frames
        for f in frames:
            try:
                # skip main frame
                if f == page.main_frame: continue
                # try find a canvas inside frame
                el = f.query_selector("canvas")
                if el:
                    try:
                        b = el.screenshot()
                        log("Captured iframe canvas screenshot")
                        return b
                    except:
                        pass
                # else screenshot frame
                # Playwright doesn't have direct frame.screenshot in sync API; fallback to full page
            except Exception:
                pass
    except Exception as e:
        log(f"find_game_screenshot error: {e}")
    # fallback full page
    try:
        b = page.screenshot(full_page=True)
        log("Captured full page screenshot")
        return b
    except Exception as e:
        log(f"Full page screenshot failed: {e}")
        return None

# ---------- main ----------
def main():
    st = github_get_state()
    sha = st.get("sha")
    state = st.get("data") or {}
    active = state.get("active", False)
    active_since = state.get("active_since")
    cooldown_until = int(state.get("cooldown_until", 0) or 0)
    now_ts = int(time.time()*1000)
    log(f"Start run prev active={active} since={active_since} cooldown_until={cooldown_until}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1280,"height":800})
        page = context.new_page()
        opened = False
        for url in DG_URLS:
            try:
                log(f"Opening {url}")
                page.goto(url, timeout=30000)
                # wait a bit
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except:
                    page.wait_for_timeout(1500)
                attempt_enter_and_wait(page)
                # allow dynamic content to render
                page.wait_for_timeout(2500)
                opened = True
                break
            except Exception as e:
                log(f"Open url error {url}: {e}")
                continue
        if not opened:
            log("Cannot open DG URLs; exiting")
            browser.close()
            return

        img_bytes = find_game_screenshot_bytes(page)
        if not img_bytes:
            log("No screenshot captured")
            browser.close()
            return

        analysis = analyze_image_bytes(img_bytes)
        summary = analysis.get("summary", {})
        log(f"Analysis summary: {summary}")

        # debug: send screenshot + summary to Telegram for confirmation
        if DEBUG == "1":
            cap = summary.copy()
            cap_text = f"[DEBUG] summary: {cap}"
            send_telegram_photo(img_bytes, caption=cap_text)
            # also send a plain text with more detail
            rows = []
            boards = analysis.get("boards", [])
            for b in boards:
                rows.append(f"Board {b['cluster']}: max_run={b['max_run']} cat={b['category']} count={b['count']}")
            send_telegram("\n".join([cap_text] + rows[:40]))

        long_count = summary.get("long_count",0)
        super_long = summary.get("super_long_count",0)
        longish = summary.get("longish_count",0)
        total_clusters = summary.get("board_clusters",0)

        overall = "胜率中等（平台收割中等时段）"
        if long_count >= MIN_BOARDS_FOR_POW:
            overall = "放水时段（提高胜率）"
        elif long_count >= MID_LONG_REQ and longish>0:
            overall = "中等胜率（中上）"
        else:
            sparse = sum(1 for b in analysis.get("boards",[]) if b["count"] < 6)
            if total_clusters>0 and sparse >= total_clusters*0.6:
                overall = "胜率调低（平台收割时段）"
            else:
                overall = "胜率中等（平台收割中等时段）"
        log(f"Overall => {overall}")

        now_iso = datetime.now(TZ).isoformat()
        in_cooldown = now_ts < cooldown_until

        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            if state.get("active", False):
                # still active -> update last_seen
                state.update({"active": True, "active_since": state.get("active_since"), "last_seen": now_iso})
                github_put_state(state, sha=sha, message="update active seen")
                log("Still active -> no new notify")
            else:
                if in_cooldown:
                    log("In cooldown, skipping notify")
                else:
                    text = f"[DG提醒] 发现放水/中上局势\n判定: {overall}\n长龙(>=8) 桌数: {long_count}\n超长龙(>=10): {super_long}\n时间: {now_iso}"
                    send_telegram(text)
                    state = {"active": True, "active_since": now_iso, "last_seen": now_iso, "cooldown_until": 0}
                    github_put_state(state, sha=sha, message="start active")
        else:
            if state.get("active", False):
                # ended
                try:
                    start_dt = datetime.fromisoformat(state.get("active_since"))
                    end_dt = datetime.now(TZ)
                    dur_min = int((end_dt - start_dt).total_seconds() / 60)
                    text = f"[DG提醒] 放水已结束\n开始: {state.get('active_since')}\n结束: {end_dt.isoformat()}\n共持续: {dur_min} 分钟"
                    send_telegram(text)
                except Exception as e:
                    log(f"duration compute error: {e}")
                cooldown_ms = COOLDOWN_MINUTES * 60 * 1000
                new_state = {"active": False, "active_since": None, "last_seen": now_iso, "cooldown_until": int(time.time()*1000) + cooldown_ms}
                github_put_state(new_state, sha=sha, message="end active")
            else:
                log("No active; nothing to do.")

        browser.close()

if __name__ == "__main__":
    main()

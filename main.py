# main.py (改良版，替换你现有的 main.py)
import os, json, time, base64, sys, traceback
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from playwright.sync_api import sync_playwright

# ========== 配置（通过 env / secrets 注入） ==========
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT  = os.environ.get("TG_CHAT")
DG_URLS = [ os.environ.get("DG_URL1","https://dg18.co/wap/"), os.environ.get("DG_URL2","https://dg18.co/") ]

MIN_BOARDS_FOR_POW = int(os.environ.get("MIN_BOARDS_FOR_POW","3"))   # 放水触发的长龙最少桌数（>=8）
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))              # 中等(中上)触发 >=2 张长龙
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))
MIN_BLOB_SIZE = int(os.environ.get("MIN_BLOB_SIZE","6"))            # blob 最小像素数
STATE_PATH = "state.json"
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
TZ = timezone(timedelta(hours=8))  # Malaysia

# ========== logger ==========
def log(msg):
    print(f"[DGMON] {datetime.now(TZ).isoformat()} - {msg}")

# ========== GitHub state helpers ==========
def github_get_state():
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return {"sha":None, "data":{}}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            j = r.json()
            return {"sha": j.get("sha"), "data": json.loads(base64.b64decode(j.get("content","")).decode())}
    except Exception as e:
        log(f"GitHub get state error: {e}")
    return {"sha": None, "data": {}}

def github_put_state(data, sha=None, message="update state"):
    if not GITHUB_REPO or not GITHUB_TOKEN:
        log("GITHUB not configured; skip save.")
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{STATE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept":"application/vnd.github+json"}
    payload = {
        "message": message,
        "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode(),
        "branch": os.environ.get("GITHUB_REF","main")
    }
    if sha: payload["sha"]=sha
    try:
        r = requests.put(url, headers=headers, json=payload, timeout=20)
        if r.status_code in (200,201):
            log("Saved state.json to repo.")
            return True
        else:
            log(f"GitHub save state failed: {r.status_code} {r.text}")
    except Exception as e:
        log(f"GitHub put state exception: {e}")
    return False

# ========== Telegram ==========
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("TG_TOKEN or TG_CHAT not set; skipping telegram.")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id":TG_CHAT, "text":text}, timeout=15)
        if r.status_code == 200:
            log("Telegram sent")
            return True
        else:
            log(f"Telegram send failed: {r.status_code} {r.text}")
            return False
    except Exception as e:
        log(f"Telegram exception: {e}")
        return False

# ========== 图像分析核心（更稳健） ==========
def analyze_image_bytes(img_bytes):
    try:
        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        log(f"Open image error: {e}")
        return {"boards":[], "summary":{"total_blobs":0}}

    np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h,w = np_img.shape[:2]
    # 缩放以限制计算量
    scale = 1.0
    max_side = max(h,w)
    if max_side > 1600:
        scale = 1600.0 / max_side
        np_img = cv2.resize(np_img, (int(w*scale), int(h*scale)))
        h,w = np_img.shape[:2]

    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)

    # 更稳健的蓝/红色阈值
    lower_blue = np.array([85,50,50]); upper_blue = np.array([140,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0,50,50]); upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,50,50]); upper_red2 = np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    # 提取连通块（分别计算）
    def extract_centroids(mask):
        num, labels, stats, cents = cv2.connectedComponentsWithStats((mask>0).astype("uint8")*255)
        centers = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_BLOB_SIZE: continue
            cx = int(cents[i][0]); cy = int(cents[i][1])
            # bounds check
            cx = min(max(0,cx), w-1); cy = min(max(0,cy), h-1)
            centers.append((cx,cy,area))
        return centers

    blue_centers = extract_centroids(mask_blue)
    red_centers = extract_centroids(mask_red)

    total_blue = len(blue_centers)
    total_red  = len(red_centers)
    total_blobs = total_blue + total_red

    # 如果没有检测到任何 blob，直接返回（便于 debug）
    if total_blobs == 0:
        return {"boards":[], "summary":{"total_blobs":0, "total_red":0, "total_blue":0}}

    # 合并所有中心用于空间聚类（DBSCAN）
    pts = []
    colors = []
    for (x,y,a) in blue_centers:
        pts.append([x,y]); colors.append('P')
    for (x,y,a) in red_centers:
        pts.append([x,y]); colors.append('B')
    pts = np.array(pts)

    # adaptive eps: proportion to image width
    eps = max(40, int(w / 12))
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=1).fit(pts)
        labels = db.labels_
    except Exception as e:
        log(f"DBSCAN failed: {e}")
        labels = np.zeros(len(pts), dtype=int)

    clusters = {}
    for i, lb in enumerate(labels):
        clusters.setdefault(int(lb), []).append({"x":int(pts[i,0]), "y":int(pts[i,1]), "color": colors[i]})

    boards = []
    long_count = 0; super_long_count = 0; longish_count = 0

    # column grouping threshold proportional to width
    col_gap_thresh = max(18, int(w / 80))

    for cid, items in clusters.items():
        # sort by x then y
        items_sorted = sorted(items, key=lambda it:(it["x"], it["y"]))
        cols = []
        for it in items_sorted:
            if not cols:
                cols.append([it])
            else:
                # if x is close to last column's last x -> same column
                if abs(it["x"] - cols[-1][-1]["x"]) <= col_gap_thresh:
                    cols[-1].append(it)
                else:
                    cols.append([it])
        seq = []
        for col in cols:
            col_sorted = sorted(col, key=lambda p:p["y"])
            seq.extend([p["color"] for p in col_sorted])
        # compute runs
        runs = []
        if seq:
            cur = seq[0]; ln = 1
            for s in seq[1:]:
                if s == cur:
                    ln += 1
                else:
                    runs.append((cur, ln)); cur = s; ln = 1
            runs.append((cur, ln))
        max_run = max([r[1] for r in runs]) if runs else 0
        cat='other'
        if max_run >= 10:
            cat='super_long'; super_long_count += 1; long_count += 1
        elif max_run >= 8:
            cat='long'; long_count += 1
        elif max_run >= 4:
            cat='longish'; longish_count += 1
        boards.append({"cluster":cid, "count": len(items), "max_run": max_run, "category": cat, "runs": runs})

    summary = {
        "total_blobs": total_blobs,
        "total_red": total_red,
        "total_blue": total_blue,
        "board_clusters": len(boards),
        "long_count": long_count,
        "super_long_count": super_long_count,
        "longish_count": longish_count,
        "eps": eps,
        "col_gap_thresh": col_gap_thresh,
        "scale": scale,
        "image_w": w, "image_h": h
    }
    return {"boards": boards, "summary": summary}

# ========== Page enter helper ==========
def attempt_enter_and_slider(page):
    texts = ["Free", "免费试玩", "免费", "Free Play", "试玩"]
    for t in texts:
        try:
            el = page.query_selector(f"text={t}")
            if el:
                try:
                    el.click(timeout=3000)
                    page.wait_for_timeout(1200)
                    return True
                except:
                    pass
        except: pass
    # try slider-like element
    try:
        el = page.query_selector("[role=slider], .slider, .drag, .verify-slider, .slide-btn")
        if el:
            box = el.bounding_box()
            if box:
                sx = box["x"] + 2; sy = box["y"] + box["height"]/2
                ex = box["x"] + box["width"] - 4
                page.mouse.move(sx, sy); page.mouse.down()
                page.mouse.move(ex, sy, steps=18); page.mouse.up()
                page.wait_for_timeout(800)
                return True
    except Exception as e:
        log(f"slider attempt error: {e}")
    return False

# ========== capture & decide ==========
def capture_and_analyze(page):
    img_bytes = page.screenshot(full_page=True)
    analysis = analyze_image_bytes(img_bytes)
    return analysis, img_bytes

def main_run():
    st = github_get_state()
    sha = st.get("sha")
    state = st.get("data") or {}
    active = state.get("active", False)
    active_since = state.get("active_since")
    cooldown_until = int(state.get("cooldown_until", 0) or 0)
    now_ms = int(time.time()*1000)
    log(f"Start run. prev active={active} since={active_since} cooldown_until={cooldown_until}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1280,"height":800})
        page = context.new_page()
        opened = False
        for url in DG_URLS:
            try:
                log(f"Opening {url}")
                page.goto(url, timeout=30000)
                page.wait_for_timeout(1200)
                attempt_enter_and_slider(page)
                page.wait_for_timeout(1600)
                opened = True
                break
            except Exception as e:
                log(f"Open error {url}: {e}")
                continue
        if not opened:
            browser.close(); log("Unable to open DG urls. Exit run."); return

        analysis, img_bytes = capture_and_analyze(page)
        summary = analysis.get("summary", {})
        log(f"Analysis summary: {summary}")

        long_count = int(summary.get("long_count",0))
        super_long = int(summary.get("super_long_count",0))
        longish = int(summary.get("longish_count",0))
        total_clusters = int(summary.get("board_clusters",0))
        total_blobs = int(summary.get("total_blobs",0))

        # 判定逻辑（严格按你说的规则）
        overall = "胜率中等（平台收割中等时段）"
        # 放水时段：至少 MIN_BOARDS_FOR_POW 张为长龙或超长龙
        if long_count >= MIN_BOARDS_FOR_POW:
            overall = "放水时段（提高胜率）"
        elif long_count >= MID_LONG_REQ and longish > 0:
            overall = "中等胜率（中上）"
        else:
            # sparse detection -> 胜率调低/收割
            sparse_boards = sum(1 for b in analysis.get("boards",[]) if b.get("count",0) < 6)
            if total_clusters>0 and sparse_boards >= total_clusters*0.6:
                overall = "胜率调低（平台收割时段）"
            else:
                overall = "胜率中等（平台收割中等时段）"

        log(f"Overall classification: {overall}  (long_count={long_count}, super_long={super_long}, longish={longish}, clusters={total_clusters}, blobs={total_blobs})")

        now_iso = datetime.now(TZ).isoformat()
        in_cooldown = now_ms < int(cooldown_until or 0)

        # notify / state update
        if overall in ("放水时段（提高胜率）","中等胜率（中上）"):
            if state.get("active", False):
                # still active -> update last_seen
                state.update({"active": True, "active_since": state.get("active_since"), "last_seen": now_iso})
                github_put_state(state, sha=sha, message="update active seen")
                log("Still active -> no notify")
            else:
                if in_cooldown:
                    log("In cooldown -> skipping notify")
                else:
                    text = (f"[DG提醒] 发现放水/中上局势\n判定: {overall}\n长龙(>=8) 桌数: {long_count}\n超长龙(>=10): {super_long}\n总点数: {total_blobs}\n时间: {now_iso}\n备注: 按设定阈值触发。")
                    send_telegram(text)
                    state = {"active": True, "active_since": now_iso, "last_seen": now_iso, "cooldown_until": 0}
                    github_put_state(state, sha=sha, message="start active")
        else:
            if state.get("active", False):
                # just ended
                try:
                    start_dt = datetime.fromisoformat(state.get("active_since"))
                    end_dt = datetime.now(TZ)
                    dur_min = int((end_dt - start_dt).total_seconds()/60)
                    text = (f"[DG提醒] 放水已结束\n开始: {state.get('active_since')}\n结束: {end_dt.isoformat()}\n共持续: {dur_min} 分钟")
                    send_telegram(text)
                except Exception as e:
                    log(f"End notify compute error: {e}")
                cooldown_ms = COOLDOWN_MINUTES * 60 * 1000
                new_state = {"active": False, "active_since": None, "last_seen": now_iso, "cooldown_until": int(time.time()*1000)+cooldown_ms}
                github_put_state(new_state, sha=sha, message="end active")
            else:
                log("No active to end; nothing to do.")

        browser.close()
        log("Run finished.")

# ========== 主入口，保证不抛异常导致非0退出 ==========
if __name__ == "__main__":
    try:
        main_run()
    except Exception as e:
        # 捕获所有异常，输出 trace，仍然优雅退出（exit 0）
        log("Uncaught exception in main_run: " + str(e))
        traceback.print_exc()
    finally:
        # Ensure process exits 0 to avoid non-zero exit codes in Actions logs
        sys.exit(0)

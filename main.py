# main.py
import os, sys, json, time, math, subprocess
from datetime import datetime, timezone
import requests
import numpy as np
import cv2

# playright
from playwright.sync_api import sync_playwright

REPO_STATE_FILE = "state.json"
# ---------- Configurable thresholds (may need tuning) ----------
COLOR_MIN_AREA = 40               # circle minimal area px
TABLE_CLUSTER_GAP_X = 200         # px gap to split tables horizontally (tune per screenshot width)
ROW_CLUSTER_DIST = 18             # px threshold to cluster dots into rows
COL_CLUSTER_DIST = 18             # px threshold to cluster dots into cols
LONG_RUN_THRESHOLD = 4            # >=4 => 长连
DRAGON_THRESHOLD = 8              # >=8 => 长龙
SUPER_DRAGON_THRESHOLD = 10       # >=10 => 超长龙
ALTERNATE_SINGLE_JUMP_MIN = 4     # 连续单跳 >=4 则视为“单跳连续”
# ----------------------------------------------------------------

TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")
DG_URLS = os.getenv("DG_URLS", "https://dg18.co,https://dg18.co/wap/")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

if not TG_TOKEN or not TG_CHAT_ID:
    print("ERROR: TG_TOKEN or TG_CHAT_ID missing. Set GitHub secrets.")
    sys.exit(1)

def send_telegram(text, image_path=None):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, data=payload, timeout=15)
        # send image if exists
        if image_path and os.path.exists(image_path):
            files_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
            with open(image_path, "rb") as f:
                requests.post(files_url, data={"chat_id": TG_CHAT_ID, "caption": text}, files={"photo": f}, timeout=30)
        return resp.ok
    except Exception as e:
        print("TG send error:", e)
        return False

def load_state():
    if os.path.exists(REPO_STATE_FILE):
        try:
            with open(REPO_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"active": False, "start_iso": None}

def save_state(state):
    with open(REPO_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    # commit & push changes back to repo so next runs see the state
    try:
        subprocess.run(["git", "config", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "github-actions"], check=True)
        subprocess.run(["git", "add", REPO_STATE_FILE], check=True)
        subprocess.run(["git", "commit", "-m", "Update state.json by DG Monitor"], check=True)
        # push using token
        repo = os.environ.get("GITHUB_REPOSITORY")
        if repo and GITHUB_TOKEN:
            remote = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{repo}.git"
            subprocess.run(["git", "push", remote, "HEAD:refs/heads/" + os.environ.get("GITHUB_REF", "refs/heads/main").split("/")[-1]], check=False)
    except Exception as e:
        print("git commit/push error:", e)

# ---------- Image analysis helpers ----------
def detect_color_circles(image_path):
    """
    Return list of detected circles: [(cx, cy, color_code)]  where color_code 'R' or 'B' or 'G'
    This uses HSV thresholds to find red & blue circles in the screenshot.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # blue mask
    lower_blue = np.array([90, 50, 40])
    upper_blue = np.array([140, 255, 255])
    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    # red mask (two ranges)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_r = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # green (tie) optional
    lower_g = np.array([40, 40, 40])
    upper_g = np.array([85, 255, 255])
    mask_g = cv2.inRange(hsv, lower_g, upper_g)

    circles = []
    for mask, code in [(mask_r, 'R'), (mask_b, 'B'), (mask_g, 'G')]:
        # morphology
        kernel = np.ones((3,3), np.uint8)
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < COLOR_MIN_AREA:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            circles.append((int(x), int(y), code))
    return circles

def cluster_tables_by_x(circles):
    """Cluster centroids into tables by large gaps in x (simple heuristic)."""
    if not circles:
        return []
    xs = sorted([c[0] for c in circles])
    # sort unique positions
    xs_sorted = sorted(xs)
    groups = []
    current_group = [xs_sorted[0]]
    for i in range(1, len(xs_sorted)):
        if xs_sorted[i] - xs_sorted[i-1] > TABLE_CLUSTER_GAP_X:
            groups.append(current_group)
            current_group = [xs_sorted[i]]
        else:
            current_group.append(xs_sorted[i])
    groups.append(current_group)
    # map groups to x ranges
    table_ranges = []
    for g in groups:
        table_ranges.append((min(g)-20, max(g)+20))
    # assign circles to table index
    tables = {}
    for cx, cy, code in circles:
        for i, (xmin, xmax) in enumerate(table_ranges):
            if xmin <= cx <= xmax:
                tables.setdefault(i, []).append((cx, cy, code))
                break
    # return list of table circle lists
    return [tables[i] for i in sorted(tables.keys())]

def grid_from_table(circles):
    """
    From table circles produce grid mapping (row_index, col_index) -> color
    clustering by y then x.
    """
    if not circles:
        return [], []
    ys = sorted(set([c[1] for c in circles]))
    xs = sorted(set([c[0] for c in circles]))
    # cluster ys into rows by proximity
    row_centers = []
    for y in sorted(ys):
        if not row_centers or abs(y - row_centers[-1]) > ROW_CLUSTER_DIST:
            row_centers.append(y)
    col_centers = []
    for x in sorted(xs):
        if not col_centers or abs(x - col_centers[-1]) > COL_CLUSTER_DIST:
            col_centers.append(x)
    # build matrix rows x cols filled with None
    rows = len(row_centers)
    cols = len(col_centers)
    grid = [[None for _ in range(cols)] for __ in range(rows)]
    for cx, cy, code in circles:
        # find nearest row/col
        rid = min(range(rows), key=lambda i: abs(row_centers[i]-cy)) if rows>0 else 0
        cid = min(range(cols), key=lambda j: abs(col_centers[j]-cx)) if cols>0 else 0
        grid[rid][cid] = code
    return grid, (row_centers, col_centers)

def analyze_grid_for_runs(grid):
    """
    Analyze vertical runs (top->down per column) and detect:
    - max_run_length per color
    - whether grid shows alternating single-jump pattern >= ALTERNATE_SINGLE_JUMP_MIN
    """
    if not grid:
        return {"max_run":0, "longest_color":None, "is_alternating_long":False}
    rows = len(grid)
    cols = len(grid[0]) if rows>0 else 0
    max_run = 0
    longest_color = None
    alternating_count = 0
    for c in range(cols):
        prev = None
        run = 0
        alt_seq = []
        for r in range(rows):
            cur = grid[r][c]
            if cur is None:
                # break vertical chain
                prev = None
                run = 0
                continue
            if cur == prev:
                run += 1
            else:
                run = 1
            prev = cur
            if run > max_run:
                max_run = run
                longest_color = cur
        # check alternating along rows for this column (simple)
        col_seq = [grid[r][c] for r in range(rows) if grid[r][c] is not None]
        if len(col_seq) >= ALTERNATE_SINGLE_JUMP_MIN:
            # check pattern ABAB...
            is_alt = all(col_seq[i] != col_seq[i+1] for i in range(len(col_seq)-1))
            if is_alt and len(col_seq) >= ALTERNATE_SINGLE_JUMP_MIN:
                alternating_count += 1
    is_alternating_long = (alternating_count >= 1)
    return {"max_run": max_run, "longest_color": longest_color, "is_alternating_long": is_alternating_long}

# ---------- Main detection flow ----------
def attempt_capture_and_analyze(url, tmp_image="dg_screenshot.png"):
    print("Visiting", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = browser.new_context(viewport={"width":1400,"height":900})
        page = context.new_page()
        page.set_default_timeout(20000)
        try:
            page.goto(url)
            time.sleep(1.5)
            # try clicking text "Free" or local variants
            for q in ["text=Free", "text=Free Play", "text=免费试玩", "text=试玩", "text=Free Trial"]:
                try:
                    el = page.query_selector(q)
                    if el:
                        el.click(timeout=3000)
                        print("Clicked", q)
                        time.sleep(1.2)
                        break
                except:
                    pass
            # try small slider automation (best-effort)
            try:
                # find any element that looks like slider by id/class keywords
                handle = page.query_selector("div[class*='slider'], div[class*='geetest'], div[id*='slider']")
                if handle:
                    box = handle.bounding_box()
                    sx = box["x"] + 5
                    sy = box["y"] + box["height"]/2
                    ex = box["x"] + box["width"] - 10
                    page.mouse.move(sx, sy)
                    page.mouse.down()
                    page.mouse.move(ex, sy, steps=25)
                    page.mouse.up()
                    print("Tried slider drag")
                    time.sleep(1.2)
            except Exception as e:
                print("slider attempt err", e)
            # wait for possible lobby to load
            page.wait_for_timeout(2500)
            # take full page screenshot
            page.screenshot(path=tmp_image, full_page=True)
            print("Saved screenshot", tmp_image)
            browser.close()
            # analyze image
            circles = detect_color_circles(tmp_image)
            if not circles:
                print("No circles detected on screenshot")
                return {"ok": False, "reason": "no circles", "image": tmp_image}
            # cluster per table
            tables = cluster_tables_by_x(circles)
            results = []
            for t in tables:
                grid, centers = grid_from_table(t)
                info = analyze_grid_for_runs(grid)
                results.append(info)
            return {"ok": True, "tables": results, "image": tmp_image}
        except Exception as e:
            print("visit/analyze error", e)
            try:
                page.screenshot(path=tmp_image, full_page=True)
            except:
                pass
            browser.close()
            return {"ok": False, "reason": str(e), "image": tmp_image}

def aggregate_judgement(all_results):
    """
    all_results is list of per-URL analysis return dicts
    returns True if '放水' detected per rules.
    """
    # flatten tables
    tables = []
    for r in all_results:
        if r.get("ok") and r.get("tables"):
            tables.extend(r["tables"])
    if not tables:
        return {"is_putong": False, "reason": "no tables"}
    # exclude tables with alternating single-jump >=4
    valid_tables = [t for t in tables if not t.get("is_alternating_long")]
    # counts for dragon/long/super
    cnt_long = sum(1 for t in valid_tables if t.get("max_run",0) >= LONG_RUN_THRESHOLD)
    cnt_dragon = sum(1 for t in valid_tables if t.get("max_run",0) >= DRAGON_THRESHOLD)
    cnt_super = sum(1 for t in valid_tables if t.get("max_run",0) >= SUPER_DRAGON_THRESHOLD)
    # rule A: if >=3 tables with 龙 or 超龙
    if (cnt_dragon + cnt_super) >= 3:
        return {"is_putong": True, "mode": "龙型", "cnt_dragon": cnt_dragon, "cnt_super": cnt_super}
    # rule B: if many tables have long连 AND overall many circles => treat as 满局面型
    # heuristic: if majority tables have long连
    if len(valid_tables)>0 and (cnt_long / len(valid_tables)) >= 0.5 and len(valid_tables) >= 6:
        return {"is_putong": True, "mode": "满局面型", "cnt_long": cnt_long, "valid_tables": len(valid_tables)}
    return {"is_putong": False, "cnt_long": cnt_long, "cnt_dragon": cnt_dragon, "cnt_super": cnt_super, "valid_tables": len(valid_tables)}

def main():
    # iterate DG urls
    urls = [u.strip() for u in DG_URLS.split(",") if u.strip()]
    print("URLs:", urls)
    all_results = []
    for u in urls:
        res = attempt_capture_and_analyze(u)
        all_results.append(res)
        time.sleep(1)
    judgement = aggregate_judgement(all_results)
    print("Judgement:", judgement)
    state = load_state()
    now = datetime.now(timezone.utc)
    # if putong (放水) detected and currently not active => start alert
    if judgement.get("is_putong"):
        if not state.get("active"):
            state["active"] = True
            state["start_iso"] = now.isoformat()
            save_state(state)
            # estimate remaining minutes (heuristic)
            # use max_run across tables for rough estimate:
            max_run = 0
            for r in all_results:
                if r.get("ok") and r.get("tables"):
                    for t in r["tables"]:
                        max_run = max(max_run, t.get("max_run",0))
            est_remain = max(3, min(60, int(max(3, max_run/1.0 * 1.5))))  # heuristic
            emoji = "🟢"
            text = f"{emoji} <b>放水时段检测到</b>\n模式: {judgement.get('mode','Unknown')}\n预计剩余约: {est_remain} 分钟（启发式估计）\n开始时间(UTC+8): { (now.astimezone(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')) }\n说明: 脚本已进入监控状态，结束后会再通知并附上实际持续时间。"
            send_telegram(text, image_path=all_results[0].get("image") if all_results else None)
        else:
            print("Already active — no new notification")
    else:
        # not detected now
        if state.get("active"):
            # previously active, now ended -> compute duration and notify
            start_iso = state.get("start_iso")
            try:
                start_dt = datetime.fromisoformat(start_iso)
            except:
                start_dt = None
            duration_min = None
            if start_dt:
                duration_min = int((now - start_dt).total_seconds() / 60)
            state["active"] = False
            state["last_end_iso"] = now.isoformat()
            save_state(state)
            emoji = "🔴"
            if duration_min is None:
                txt = f"{emoji} 放水已结束（检测到结束）。持续时间: 无法确定（缺少开始时间）。"
            else:
                txt = f"{emoji} 放水已结束\n共持续: {duration_min} 分钟\n结束时间(UTC+8): { now.astimezone(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S') }"
            send_telegram(txt, image_path=all_results[0].get("image") if all_results else None)
        else:
            print("No active putong and none detected now.")
    # finish
    return

if __name__ == "__main__":
    main()

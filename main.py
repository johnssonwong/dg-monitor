# main.py
# DG 实盘监测脚本（Playwright + OpenCV）
# 设计目标：在 GitHub Actions 每 5 分钟执行一次；尽最大努力进入 DG 实盘并检测“放水 / 中等胜率（中上）”，并在触发时发送 Telegram 开始/结束通知（含估算/实际时长）。
# 注意：尽力而为，但无法保证 100% 成功（见脚本顶部说明）。
import os, sys, time, json, math, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

# Playwright
from playwright.sync_api import sync_playwright

# scikit KMeans fallback
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "").strip()
# DG links
DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW","3"))  # 放水最少合格桌数
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))             # 中等胜率需要的长龙桌数
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))    # 若触发后冷却分钟（开始后进入 cooldown until predicted end）
STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))  # Malaysia UTC+8

# ---------------- helpers ----------------
def now_ts():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(s):
    print(f"[{now_ts()}] {s}", flush=True)

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置：跳过 send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": text}
        r = requests.post(url, data=payload, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log(f"Telegram 返回错误: {j}")
            return False
    except Exception as e:
        log(f"Telegram 发送异常: {e}")
        return False

def load_state():
    if not os.path.exists(STATE_FILE):
        s = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
        return s
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ------------- image helpers -------------
def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(bgr_img):
    """
    返回点列表 (x,y,color) ， color 'B' = banker (red), 'P' = player (blue)
    使用 HSV 阈值检测红/蓝点，并做简单形态学去噪。
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # red ranges
    lower1 = np.array([0,120,60]); upper1 = np.array([10,255,255])
    lower2 = np.array([160,120,60]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb = np.array([90,60,40]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    pts = []
    def contours_to_centers(mask, label):
        ctrs,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in ctrs:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if M["m00"]==0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            pts.append((cx, cy, label))
    contours_to_centers(mask_r, 'B')
    contours_to_centers(mask_b, 'P')
    return pts, mask_r, mask_b

def cluster_boards(points, img_w, img_h):
    """
    将点聚成若干桌子区域；优先用网格密度法，失败时用 KMeans。
    返回 list of rects (x,y,w,h)
    """
    if not points:
        return []
    cell = max(48, int(min(img_w, img_h)/14))
    cols = math.ceil(img_w / cell)
    rows = math.ceil(img_h / cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    thr = 6  # 单元阈值
    hits = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]>=thr]
    rects = []
    if hits:
        for (r,c) in hits:
            x = c*cell; y = r*cell; w = cell; h = cell
            merged=False
            for idx,(rx,ry,rw,rh) in enumerate(rects):
                if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                    nx = min(rx,x); ny = min(ry,y); nw = max(rx+rw, x+w)-nx; nh = max(ry+rh, y+h)-ny
                    rects[idx] = (nx,ny,nw,nh); merged=True; break
            if not merged:
                rects.append((x,y,w,h))
        # expand a bit
        regs = []
        for (x,y,w,h) in rects:
            nx = max(0, x-10); ny = max(0,y-10); nw = min(img_w-nx, w+20); nh = min(img_h-ny, h+20)
            regs.append((int(nx),int(ny),int(nw),int(nh)))
        return regs
    # fallback KMeans
    pts_arr = np.array([[p[0],p[1]] for p in points])
    k = min(6, max(1, len(points)//10))
    if k<=0:
        return []
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pts_arr)
    regs=[]
    for lab in range(k):
        sel = pts_arr[kmeans.labels_==lab]
        if sel.shape[0]==0: continue
        x0,y0 = sel.min(axis=0); x1,y1 = sel.max(axis=0)
        nx,ny = max(0,int(x0-12)), max(0,int(y0-12))
        nw,nh = min(img_w-nx, int(x1-x0+24)), min(img_h-ny, int(y1-y0+24))
        regs.append((nx,ny,nw,nh))
    return regs

def analyze_board(img_bgr, rect):
    x,y,w,h = rect
    crop = img_bgr[y:y+h, x:x+w]
    pts,_,_ = detect_red_blue_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "runs":[], "flattened":[]}
    # cluster by column using x coordinate
    pts_local = [(px,py,c) for (px,py,c) in pts]
    xs = [p[0] for p in pts_local]
    # heuristic group columns
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups = []
    for i in sorted_idx:
        xval = xs[i]
        placed=False
        for grp in col_groups:
            meanx = sum([pts_local[j][0] for j in grp])/len(grp)
            if abs(meanx - xval) <= max(8, w//45):
                grp.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    sequences=[]
    for grp in col_groups:
        col_pts = sorted([pts_local[i] for i in grp], key=lambda t: t[1])
        sequences.append([p[2] for p in col_pts])
    # flatten reading column by column top->bottom
    flattened=[]
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
    # compute runs
    runs=[]
    if flattened:
        cur = {"color": flattened[0], "len":1}
        for i in range(1, len(flattened)):
            if flattened[i]==cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur)
                cur = {"color":flattened[i], "len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    # detect "multi-row 连珠/multi" heuristic:
    # count columns with local max run >=4
    multi_cols = sum(1 for col in sequences if any(run_len>=4 for run_len in [len([c for c in col if c == col[0]])])) if sequences else 0
    # simpler: check if sequences has at least 3 columns with length>=4 (heuristic for 连珠/多连)
    multi_cols2 = sum(1 for col in sequences if len(col) >= 4)
    is_multi = multi_cols2 >= 3
    return {"total": len(flattened), "maxRun": maxRun, "category": cat, "runs": runs, "flattened": flattened, "is_multi": is_multi, "multi_cols": multi_cols2}

def classify_all(board_stats):
    longCount = sum(1 for b in board_stats if b.get("category") in ("long","super_long"))
    superCount = sum(1 for b in board_stats if b.get("category")=="super_long")
    # 中等胜率（中上）判定（你要求）：
    # 至少 3 张桌子有 连续3排“多连/连珠”(我们用 is_multi heuristic)，并且至少 2 张桌子是 龙头/超长龙（可与多连同一桌）
    multi_count = sum(1 for b in board_stats if b.get("is_multi"))
    longishCount = sum(1 for b in board_stats if b.get("category") in ("long","super_long"))
    # 判定
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount
    if multi_count >= 3 and longishCount >= 2:
        return "中等胜率（中上）", longCount, superCount
    # 收割 / 胜率中等
    sparse = sum(1 for b in board_stats if b.get("total",0) < 6)
    n = max(1, len(board_stats))
    if sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount
    return "胜率中等（平台收割中等时段）", longCount, superCount

# -------------- Playwright + Capture --------------
def attempt_enter_and_screenshot(play, url, tries=2):
    browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage"])
    screenshot = None
    try:
        context = browser.new_context(viewport={"width":1280,"height":800}, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36")
        page = context.new_page()
        log(f"打开 URL: {url}")
        page.goto(url, timeout=40000)
        time.sleep(1.2)
        # 尝试点击 Free / 免费 / Play Free 等
        clicked=False
        btn_texts = ["Free", "免费试玩", "免费", "Play Free", "试玩", "Start"]
        for txt in btn_texts:
            try:
                el = page.locator(f"text={txt}")
                if el.count()>0:
                    el.first.click(timeout=5000)
                    clicked=True
                    log(f"点击文本按钮: {txt}")
                    break
            except Exception:
                pass
        time.sleep(1.2)
        # 尝试寻找滑动安全条（多种策略）
        # 1) 寻找 input[type=range] 并设置 value
        try:
            el = page.query_selector("input[type=range]")
            if el:
                page.evaluate("(el)=>el.value=el.max", el)
                log("找到 input range，设置为 max.")
                time.sleep(1)
        except Exception:
            pass
        # 2) 尝试查找常见滑块类名/元素并用鼠标模拟拖动
        slider_selectors = [
            "div[class*=slider]", "div[class*=drag]", "div[id*=slider]", "div[class*=verify]", "div[class*=captcha]", "div[role='slider']"
        ]
        dragged=False
        for sel in slider_selectors:
            try:
                items = page.query_selector_all(sel)
                if items and len(items)>0:
                    for it in items:
                        try:
                            box = it.bounding_box()
                            if box and box["width"]>20:
                                # 模拟从左到右拖动
                                sx = box["x"]+5; sy = box["y"]+box["height"]/2
                                ex = box["x"]+box["width"]-6
                                page.mouse.move(sx, sy); page.mouse.down()
                                steps = 26
                                for s in range(steps):
                                    nx = sx + (ex - sx)*(s+1)/steps
                                    page.mouse.move(nx, sy, steps=1)
                                    time.sleep(0.02)
                                page.mouse.up()
                                dragged=True
                                log(f"尝试拖动滑块（selector {sel}).")
                                time.sleep(1.2)
                                break
                        except Exception:
                            continue
                if dragged: break
            except Exception:
                continue
        # 3) 如果仍然未被动，通过滚动页面来触发“安全条完成”
        try:
            for _ in range(6):
                page.mouse.wheel(0, 400)
                time.sleep(0.35)
            time.sleep(1.0)
        except Exception:
            pass
        # 等候实盘区域加载，检测页面是否包含大量珠点图（红/蓝）
        time.sleep(3.5)
        # 最后截视图（full_page 可能失败在动态内容），先尝试 viewport capture
        try:
            screenshot = page.screenshot(full_page=False)
            log("已截取视口截图。")
        except Exception:
            try:
                screenshot = page.screenshot(full_page=True)
                log("已截取整页截图。")
            except Exception as e:
                log(f"截图失败: {e}")
        try:
            context.close()
        except Exception:
            pass
    finally:
        try:
            browser.close()
        except Exception:
            pass
    return screenshot

# -------------- main logic --------------
def main():
    log("开始检测循环。")
    state = load_state()
    # 如果 state active 且存在一个预测结束时间且未到时间，则直接跳过检测（以达到“提醒后暂停检测直到预计结束”的需求）
    # 我们把 cooldown 存在 state e.g. state['cooldown_until'] = iso string
    cd_until = state.get("cooldown_until")
    if cd_until:
        try:
            cd_dt = datetime.fromisoformat(cd_until)
            if datetime.now(TZ) < cd_dt:
                log(f"处于提醒后冷却期，直到 {cd_dt.isoformat()} 才恢复检测。退出本次 run。")
                return
        except Exception:
            pass

    screenshot = None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                screenshot = attempt_enter_and_screenshot(p, url)
                if screenshot:
                    break
            except Exception as e:
                log(f"访问 {url} 时异常: {e}\n{traceback.format_exc()}")
                continue
    if not screenshot:
        log("无法取得有效截图，本次 run 结束。")
        # 保存 state （无变更）
        save_state(state)
        return

    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    pts, _, _ = detect_red_blue_points(bgr)
    log(f"检测到点数: {len(pts)}")
    if len(pts) < 8:
        log("检测到点数过少，可能尚未成功进入实盘或页面布局不同。保存 summary 并退出。")
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({"ts": now_ts(), "note":"low_points", "points": len(pts)}, f, ensure_ascii=False, indent=2)
        save_state(state)
        return

    regions = cluster_boards(pts, w, h)
    log(f"聚类得到候选桌数: {len(regions)}")
    board_stats=[]
    for r in regions:
        st = analyze_board(bgr, r)
        board_stats.append(st)
    overall, longCount, superCount = classify_all(board_stats)
    log(f"判定 -> {overall} (长龙桌数={longCount}, 超长龙={superCount})")
    # 保存 summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump({"ts": now_ts(), "overall": overall, "longCount": longCount, "superCount": superCount, "boards": board_stats[:30]}, f, ensure_ascii=False, indent=2)

    # 状态机（start / ongoing / end）
    was_active = state.get("active", False)
    was_kind = state.get("kind")
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    if is_active_now and not was_active:
        # 开始新的事件
        start_time = datetime.now(TZ)
        # 估算结束时间：从历史平均取
        history = state.get("history", [])
        durations = [h.get("duration_minutes") for h in history if h.get("duration_minutes",0)>0]
        if durations:
            est_minutes = round(sum(durations)/len(durations))
            if est_minutes < 3: est_minutes = 3
        else:
            est_minutes = 10
        est_end = start_time + timedelta(minutes=est_minutes)
        # 设置 cooldown_until = est_end （在此期间我们停止检测）
        state = {"active": True, "kind": overall, "start_time": start_time.isoformat(), "last_seen": start_time.isoformat(), "history": history, "cooldown_until": est_end.isoformat()}
        save_state(state)
        msg = f"🔔 [DG提醒] {overall} 已开始。\n时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n长龙桌数: {longCount}，超长龙: {superCount}\n估计结束时间: {est_end.strftime('%Y-%m-%d %H:%M:%S')}（约 {est_minutes} 分钟）\n说明：此为基于历史估计，实际结束将再通知。"
        send_telegram(msg)
        log("发送开始提醒并进入冷却直到估计结束时间。")
        # 保存 state
        save_state(state)
        return

    if is_active_now and was_active:
        # 活动中，更新 last_seen（但我们已在开始时设了 cooldown_until）
        state["last_seen"] = datetime.now(TZ).isoformat()
        state["kind"] = overall
        save_state(state)
        log("活动仍在继续，更新 last_seen。")
        return

    if not is_active_now and was_active:
        # 事件结束（我们可能在冷却期到期后再次检测到非活动）
        try:
            start = datetime.fromisoformat(state.get("start_time"))
        except Exception:
            start = datetime.now(TZ)
        end = datetime.now(TZ)
        duration_minutes = round((end - start).total_seconds() / 60)
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_minutes})
        history = history[-120:]
        state_new = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history, "cooldown_until": None}
        save_state(state_new)
        msg = f"✅ [DG提醒] {state.get('kind')} 已结束。\n开始: {start.strftime('%Y-%m-%d %H:%M:%S')}\n结束: {end.strftime('%Y-%m-%d %H:%M:%S')}\n实际持续: {duration_minutes} 分钟。"
        send_telegram(msg)
        log("活动结束通知已发送并记录历史。")
        return

    # else not active and was not active
    save_state(state)
    log("目前未处于放水或中上时段，未发送提醒。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"脚本执行异常: {e}\n{traceback.format_exc()}")
        # 保证异常时保存最少的状态
        st = load_state()
        save_state(st)
        sys.exit(1)

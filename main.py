# -*- coding: utf-8 -*-
"""
DG 实盘检测器 — GitHub Actions 版（每5分钟运行一次）
功能：
 - 使用 Playwright 自动进入 DG 页面 (尝试两个入口)
 - 模拟点击 Free / 免费试玩、模拟滚动/拖动滑块（多次尝试）
 - 截图并使用 OpenCV/NumPy/Scikit-learn 分析每桌珠子分布
 - 严格按用户规则判断：
    * 长连 >=4 (longish)
    * 龙 = 连续 >=8 (long)
    * 超龙 = 连续 >=10 (super_long)
    * 连珠/多连：同一行出现连续 >=4 的横向连 (horizontal run >=4)
    * 连续3排多连：检测到任意 3 个**连续行**（row r,r+1,r+2）每行均有横向连 >=4
 - 判定总体：
    * 放水时段：满足 超长龙+2长龙（超龙>=1 and 长龙>=2 and 总 >=3） OR 长龙/超龙的桌数 >= MIN_BOARDS_FOR_PAW
    * 中等胜率（中上）：至少 3 张桌子有“连续3排多连” 且 至少 2 张桌子为长龙/超长龙（可以为同一桌）
    * 其余判定为胜率中等或胜率调低 （按稀疏度判断）
 - 当进入 放水 或 中等（中上）时发送 Telegram 开始通知（含估算结束时间基于历史平均），并进入活动状态（不会重复提醒）
 - 当活动结束时发送 Telegram 结束通知（含真实持续分钟数），并保存历史
 - 输出 last_run_summary.json 供调试
"""

import os, sys, time, json, math, random
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path

# image libs
import cv2
from PIL import Image

# clustering
from sklearn.cluster import KMeans

# playwright
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# config / env
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT  = os.environ.get("TG_CHAT_ID", "").strip()
DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW","3"))
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ","2"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES","10"))

STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"
TZ = timezone(timedelta(hours=8))

def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# Telegram helper
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        log("Telegram 未配置（TG_BOT_TOKEN 或 TG_CHAT_ID 为空），跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id":TG_CHAT, "text": text, "parse_mode":"HTML"}
    try:
        r = requests.post(url, data=payload, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log(f"Telegram API 返回: {j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
        return False

# state
def load_state():
    if not os.path.exists(STATE_FILE):
        s = {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}
        return s
    try:
        with open(STATE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"读取 state.json 出错: {e}")
        return {"active":False,"kind":None,"start_time":None,"last_seen":None,"history":[]}

def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# image utilities
def pil_from_bytes(bts):
    return Image.open(BytesIO(bts)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# detect red and blue bead centers robustly
def detect_beads(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # red thresholds (two ranges)
    lower1 = np.array([0,100,70]); upper1 = np.array([8,255,255])
    lower2 = np.array([160,80,70]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue
    lowerb = np.array([90,60,50]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)

    # clean
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)

    points = []
    # find centers using contours
    for mask, label in [(mask_r,'B'), (mask_b,'P')]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8: continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

# cluster points into board regions (grid-based + merge)
def cluster_boards(points, w, h):
    if not points:
        return []
    # coarse cell size derived from image size
    cell = max(60, int(min(w,h)/12))
    cols = math.ceil(w/cell); rows = math.ceil(h/cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell); cy = min(rows-1, y//cell)
        grid[cy][cx] += 1
    hits=[]
    thr = max(3, int(len(points)/(6*max(1,min(cols,rows)))))  # adaptive threshold
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr:
                hits.append((r,c))
    if not hits:
        # fallback KMeans to cluster into up to 8 regions
        coords = np.array([[p[0], p[1]] for p in points], dtype=float)
        k = min(8, max(1, len(points)//15))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)
        regions=[]
        for lab in range(k):
            pts = coords[kmeans.labels_==lab]
            if len(pts)==0: continue
            x0,y0 = pts.min(axis=0); x1,y1 = pts.max(axis=0)
            regions.append((int(max(0,x0-8)), int(max(0,y0-8)), int(min(w, x1-x0+16)), int(min(h, y1-y0+16))))
        return regions
    rects=[]
    for (r,c) in hits:
        x = c*cell; y = r*cell; wcell = cell; hcell = cell
        placed=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+wcell < rx-cell or y > ry+rh+cell or y+hcell < ry-cell):
                nx = min(rx,x); ny = min(ry,y)
                nw = max(rx+rw, x+wcell) - nx
                nh = max(ry+rh, y+hcell) - ny
                rects[i] = (nx,ny,nw,nh)
                placed=True; break
        if not placed:
            rects.append((x,y,wcell,hcell))
    regions=[]
    for (x,y,w0,h0) in rects:
        nx=max(0,x-10); ny=max(0,y-10)
        nw=min(w-nx, w0+20); nh=min(h-ny, h0+20)
        regions.append((int(nx),int(ny),int(nw),int(nh)))
    return regions

# analyze single board region: build matrix of rows x cols, compute runs and horizontal runs
def analyze_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts, _, _ = detect_beads(crop)
    if not pts:
        return {"total":0,"maxRun":0,"category":"empty","has_multirow":False,"runs":[],"grid":None}
    # positions
    coords = np.array([[p[0], p[1]] for p in pts])
    labels = [p[2] for p in pts]
    # estimate number of columns: try kmeans on x with k up to 12
    est_cols = min(18, max(3, int(w / max(20, w//12))))
    # try multiple k to find stable clustering using inertia heuristic
    best_k = min(est_cols, max(3, len(coords)//6))
    if len(coords) < 8:
        best_k = max(1, len(coords)//2)
    # if few points, fallback to simple column grouping by binning
    if len(coords) < 6:
        # bin by x into ~5 bins
        bins = max(1, min(6, int(w/60)))
        xs = coords[:,0]
        cols_idx = np.floor(xs / (w / max(1,bins))).astype(int)
        unique_cols = sorted(set(cols_idx))
        col_positions = []
        for uc in unique_cols:
            idx = np.where(cols_idx==uc)[0]
            col_positions.append([coords[i][0] for i in idx])
        # build sequences per column
        sequences=[]
        for uc in unique_cols:
            idx = np.where(cols_idx==uc)[0]
            col_pts = sorted([(coords[i][1], labels[i]) for i in idx], key=lambda t:t[0])
            sequences.append([lab for (_,lab) in col_pts])
    else:
        # use kmeans on x to group into columns
        X = coords[:,0].reshape(-1,1)
        K = min(best_k, max(2, len(coords)//3))
        try:
            kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
            groups = [[] for _ in range(K)]
            for i,lab in enumerate(kmeans.labels_):
                groups[lab].append(i)
            # order columns by centroid x
            centroids = kmeans.cluster_centers_.flatten()
            order = sorted(range(K), key=lambda i: centroids[i])
            sequences=[]
            for oi in order:
                idxs = groups[oi]
                col_pts = sorted([(coords[i][1], labels[i]) for i in idxs], key=lambda t:t[0])
                sequences.append([lab for (_,lab) in col_pts])
        except Exception:
            # fallback to binning
            bins = max(1, min(6, int(w/60)))
            xs = coords[:,0]
            cols_idx = np.floor(xs / (w / max(1,bins))).astype(int)
            unique_cols = sorted(set(cols_idx))
            sequences=[]
            for uc in unique_cols:
                idx = np.where(cols_idx==uc)[0]
                col_pts = sorted([(coords[i][1], labels[i]) for i in idx], key=lambda t:t[0])
                sequences.append([lab for (_,lab) in col_pts])

    # flatten into bead reading order (column-major top->bottom, left->right)
    maxlen = max((len(s) for s in sequences), default=0)
    flattened=[]
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
    # compute vertical/flatten runs
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for i in range(1,len(flattened)):
            if flattened[i]==cur["color"]:
                cur["len"]+=1
            else:
                runs.append(cur); cur={"color":flattened[i],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # build approximate grid by assigning row indices from sorted unique y's per column
    # For horizontal run detection, we approximate rows by quantizing y positions across all pts
    ys = sorted(set([int(round(p[1])) for p in coords[:,1]]))
    if len(ys) == 0:
        grid = None
    else:
        # cluster y into rows using kmeans on y
        try:
            rows_k = min(len(ys), max(3, int(h/28)))
            y_coords = np.array(ys).reshape(-1,1)
            if len(y_coords) >= rows_k:
                ky = KMeans(n_clusters=rows_k, random_state=0).fit(y_coords)
                centers = sorted([c[0] for c in ky.cluster_centers_])
                # map each point to nearest center index
                row_indices = [int(np.argmin([abs(p[1]-c) for c in centers])) for p in pts]
                col_count = len(sequences)
                row_count = len(centers)
                grid = [['' for _ in range(col_count)] for __ in range(row_count)]
                # place points by nearest col index (use sequences order centroids)
                # approximate column_x positions from sequences by mean x per col
                col_xs = []
                # recover mean x of each sequence by scanning original coords mapping
                # compute col_x for each sequence by averaging xs of corresponding points
                # Here we reconstruct groups by counting sequence lengths may not give indices; fallback reasonable
                for seq in sequences:
                    # find average x among the labels corresponding - rough estimate
                    col_xs.append(np.mean([coords[i][0] for i in range(len(coords))]) if len(coords)>0 else 0)
                # instead use KMeans centroids earlier if available (we didn't save), so fallback to quantize by x bins
                xs_all = coords[:,0]
                col_bins = np.linspace(0, crop.shape[1], num=max(2,len(sequences)+1))
                for idx_pt, (px,py,lab) in enumerate(pts):
                    col_idx = np.searchsorted(col_bins, px) - 1
                    col_idx = max(0, min(len(sequences)-1, col_idx))
                    row_idx = row_indices[idx_pt]
                    grid[row_idx][col_idx] = lab
        except Exception:
            grid = None

    # check horizontal runs per row (if grid available)
    has_multirow = False
    if grid:
        # compute for each row longest horizontal same-color run
        row_runs = []
        for r in range(len(grid)):
            maxh = 0
            curc = None; curlen=0
            for c in range(len(grid[0])):
                v = grid[r][c]
                if v == curc and v != '':
                    curlen += 1
                else:
                    curc = v
                    curlen = 1 if v != '' else 0
                if curlen > maxh: maxh = curlen
            row_runs.append(maxh)
        # find any 3 consecutive rows each with horizontal run >=4
        for i in range(0, max(0,len(row_runs)-2)):
            if row_runs[i] >=4 and row_runs[i+1] >=4 and row_runs[i+2] >=4:
                has_multirow = True
                break

    # classify
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"

    return {"total":len(flattened),"maxRun":maxRun,"category":cat,"has_multirow":has_multirow,"runs":runs,"grid":grid}

# take screenshot via Playwright with robust attempts
def capture_screenshot(play, url, tries=2):
    log(f"尝试打开 {url}")
    browser = None
    try:
        browser = play.chromium.launch(headless=True, args=[
            "--no-sandbox","--disable-setuid-sandbox",
            "--disable-dev-shm-usage","--disable-accelerated-2d-canvas"
        ])
        context = browser.new_context(viewport={"width":1280,"height":900}, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0 Safari/537.36")
        page = context.new_page()
        page.set_default_timeout(35000)
        page.goto(url)
        time.sleep(2+random.random()*1.5)
        # try click common Free buttons (multi-language)
        texts = ["Free","免费试玩","免费","Play Free","试玩","进入","Free Play"]
        clicked=False
        for t in texts:
            try:
                el = page.locator(f"text={t}")
                if el.count()>0:
                    try:
                        el.first.click(timeout=3000)
                        clicked=True
                        log(f"点击按钮: {t}")
                        break
                    except Exception:
                        pass
            except Exception:
                pass
        # try to detect and handle slider or scroll security
        try:
            # scroll whole page to trigger lazy elements
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.8)
            page.evaluate("window.scrollTo(0, 0);")
            time.sleep(0.8)
            # attempt some mouse wheel actions
            for _ in range(3):
                page.mouse.wheel(0, 400)
                time.sleep(0.4)
        except Exception:
            pass
        # wait a few seconds for DG content to load
        time.sleep(3 + random.random()*1.5)
        # if there is an iframe with the table, try to screenshot entire viewport
        try:
            shot = page.screenshot(full_page=True)
            log("已截取 full_page 截图。")
            context.close()
            return shot
        except Exception:
            try:
                # fallback viewport screenshot
                shot = page.screenshot()
                context.close()
                return shot
            except Exception as e:
                log(f"截图失败: {e}")
                context.close()
                return None
    except Exception as e:
        log(f"Playwright 访问出错: {e}")
        if browser:
            try: browser.close()
            except: pass
        return None

# classify overall using the strict rules you demanded
def classify_overall(board_stats):
    long_count = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    super_count = sum(1 for b in board_stats if b['category']=='super_long')
    multirow_count = sum(1 for b in board_stats if b.get('has_multirow',False))
    # 放水：超长龙触发型 OR 满盘长连型 (这里实现两套)
    # 超长龙触发型: 至少 1 超龙 && 至少 2 长龙 && (super+long) >=3
    if super_count >= 1 and long_count >= 2 and (super_count + long_count) >= 3:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    # 满盘长连: 若满足 MIN_BOARDS_FOR_PAW 张桌是 长龙/超长龙
    if (long_count + super_count) >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", long_count, super_count, multirow_count
    # 中等胜率（中上）: 至少 3 张桌子满足 连续3排多连 && 至少 2 张为 长龙/超长龙 (可同桌)
    if multirow_count >= 3 and (long_count + super_count) >= 2:
        return "中等胜率（中上）", long_count, super_count, multirow_count
    # 若多数桌很稀疏则为 收割
    totals = [b['total'] for b in board_stats]
    sparse_count = sum(1 for t in totals if t < 6)
    if board_stats and sparse_count >= len(board_stats)*0.6:
        return "胜率调低 / 收割时段", long_count, super_count, multirow_count
    return "胜率中等（平台收割中等时段）", long_count, super_count, multirow_count

# main
def main():
    state = load_state()
    log("=== 新一次检测开始 ===")
    screenshot = None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                screenshot = capture_screenshot(p, url)
                if screenshot:
                    break
            except Exception as e:
                log(f"访问 {url} 出错: {e}")
                continue
    if not screenshot:
        log("未能获得页面截图，本次 run 结束。")
        save_state(state)
        return

    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h, w = bgr.shape[:2]
    points, mr, mb = detect_beads(bgr)
    log(f"检测到点数: {len(points)}")
    if len(points) < 8:
        log("点太少，可能页面未完全加载或选择错误（可能不是局势页面）")
    regions = cluster_boards(points, w, h)
    log(f"聚类出 {len(regions)} 个候选桌区")
    board_stats = []
    for r in regions:
        st = analyze_region(bgr, r)
        board_stats.append(st)
    # if no regions, fallback: attempt to divide whole page into grid and analyze each
    if not board_stats:
        # consider 6x4 grid
        gcols = 4; grows = 6
        wstep = w//gcols; hstep = h//grows
        for gy in range(grows):
            for gx in range(gcols):
                rx = gx*wstep; ry = gy*hstep; rw = wstep; rh = hstep
                st = analyze_region(bgr, (rx,ry,rw,rh))
                if st['total']>0:
                    board_stats.append(st)

    overall, long_count, super_count, multirow_count = classify_overall(board_stats)
    log(f"本次判定：{overall} (长龙数={long_count} 超长龙={super_count} 连续3排多连桌数={multirow_count} )")

    now = datetime.now(TZ)
    now_iso = now.isoformat()
    was_active = state.get("active", False)
    is_active = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    if is_active and not was_active:
        # 开始新事件
        history = state.get("history", [])
        est_minutes = None
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        if durations:
            est_minutes = max(1, round(sum(durations)/len(durations)))
        else:
            est_minutes = 10  # fallback
        est_end = (now + timedelta(minutes=est_minutes)).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🟢" if overall.startswith("放水") else "🔵"
        msg = f"{emoji} <b>DG 局势提醒 — {overall}</b>\n开始: {now_iso}\n长龙数: {long_count}；超长龙: {super_count}；连续3排多连桌: {multirow_count}\n估计结束: {est_end}（约 {est_minutes} 分钟，基于历史）\n\n如要手动入场，请注意风险。"
        send_telegram(msg)
        # update state
        state = {"active":True, "kind":overall, "start_time":now_iso, "last_seen":now_iso, "history": state.get("history",[])}
        save_state(state)
        log("已记录活动开始并发送通知。")

    elif is_active and was_active:
        # 持续中的活动 -> 更新 last_seen
        state["last_seen"] = now_iso
        state["kind"] = overall
        save_state(state)
        log("仍在活动中，更新 last_seen。")

    elif (not is_active) and was_active:
        # 活动结束
        start = datetime.fromisoformat(state.get("start_time"))
        end = now
        duration_minutes = round((end - start).total_seconds() / 60.0)
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_minutes})
        # cap history length
        history = history[-120:]
        # save
        new_state = {"active":False, "kind":None, "start_time":None, "last_seen":None, "history": history}
        save_state(new_state)
        msg = f"🔴 <b>DG 放水/中上 已结束</b>\n类型: {state.get('kind')}\n开始: {state.get('start_time')}\n结束: {end.isoformat()}\n实际持续: {duration_minutes} 分钟"
        send_telegram(msg)
        log("事件结束通知已发送并记录历史。")
    else:
        # not active, do nothing
        save_state(state)
        log("当前不在放水/中上时段，不发送提醒。")

    # save summary file for debugging
    summary = {"ts": now_iso, "overall":overall, "long_count":long_count, "super_count":super_count, "multirow_count":multirow_count, "boards": board_stats[:40]}
    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"未捕获异常: {e}")
        raise

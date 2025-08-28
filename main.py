# -*- coding: utf-8 -*-
"""
DG 自动检测主脚本（用于 GitHub Actions）
说明：
 - 在 Playwright 中打开 DG 链接，尝试点击 Free/免费试玩、处理滑动安全条（模拟人类拖动）
 - 截图页面并使用 OpenCV 检测红/蓝“珠子”，按用户规则计算连数（长连≥4，长龙≥8，超长龙≥10，单跳/双跳/断连开单等）
 - 根据规则判定时段：放水时段（提高胜率） / 中等胜率（中上） / 胜率中等 / 胜率调低（收割）
 - 在进入 放水 或 中等胜率 时发送 Telegram 开始提醒（含估算结束时间）；在结束时发送结束通知并记录真实持续时间
 - state.json 用于保存历史与当前活动状态（会被 workflow commit 回 repo）
注意：自动滑块与网站反自动化可能会失败；脚本内有重试、随机化移动、和降级策略。
"""

import os, sys, time, json, math, random, traceback
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------- 配置（可用 GH Secrets / env 覆盖） ----------
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "").strip()

# 两个 DG 链接（备用）
DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# 判定参数（可调整）
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))   # 放水最少满足桌数（默认 3）
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))              # 中等胜率 (中上) 需要 >=2 张长龙/超长龙
MID_MULTI_ROW_REQ = int(os.environ.get("MID_MULTI_ROW_REQ", "3"))    # “中等胜率（中上）”需 3 张桌子具有 连续3排多连/连珠（启发式检测）
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))

STATE_FILE = "state.json"
LAST_SUMMARY = "last_run_summary.json"

# 时区：马来西亚 UTC+8
TZ = timezone(timedelta(hours=8))

# ---------- 日志 ----------
def log(msg):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# ---------- Telegram ----------
def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        log("Telegram 未配置，跳过发送。")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20)
        j = r.json()
        if j.get("ok"):
            log("Telegram 发送成功。")
            return True
        else:
            log(f"Telegram 返回非 ok：{j}")
            return False
    except Exception as e:
        log(f"发送 Telegram 出错：{e}")
        return False

# ---------- state 管理 ----------
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ---------- 图像基本处理 ----------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def cv_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def detect_red_blue_points(bgr_img):
    """
    检测红/蓝珠的点位置（返回 (x,y,color) 列表），color 'B' 表示庄(红)，'P' 表示闲(蓝)。
    使用 HSV 阈值并去噪；返回 also mask images for debug.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # red mask (two ranges)
    lower1 = np.array([0,100,90]); upper1 = np.array([10,255,255])
    lower2 = np.array([160,100,90]); upper2 = np.array([179,255,255])
    mask_r = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # blue mask
    lowerb = np.array([95, 70, 50]); upperb = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lowerb, upperb)
    # morphology
    k = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k, iterations=1)
    # find contours
    points=[]
    for mask,label in [(mask_r,'B'), (mask_b,'P')]:
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            points.append((cx,cy,label))
    return points, mask_r, mask_b

def cluster_boards(points, img_w, img_h):
    """
    简单把散点聚成候选“桌子”区域（启发式），返回 region 列表 (x,y,w,h)
    """
    if not points:
        return []
    # coarse grid by cell size
    cell = max(60, int(min(img_w, img_h)/12))
    cols = math.ceil(img_w / cell); rows = math.ceil(img_h / cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,c) in points:
        cx = min(cols-1, x // cell); cy = min(rows-1, y // cell)
        grid[cy][cx] += 1
    hits=[]
    thr = max(3, int(cell/30))  # adaptive threshold
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr: hits.append((r,c))
    if not hits:
        # fallback: cluster by kmeans
        from sklearn.cluster import KMeans
        pts = np.array([[p[0],p[1]] for p in points])
        k = min(8, max(1, len(points)//8))
        try:
            km = KMeans(n_clusters=k, random_state=0).fit(pts)
            regs=[]
            for lab in range(k):
                pts_l = pts[km.labels_==lab]
                if len(pts_l)==0: continue
                x0,y0 = pts_l.min(axis=0); x1,y1 = pts_l.max(axis=0)
                regs.append((int(max(0,x0-8)), int(max(0,y0-8)), int(min(img_w, x1-x0+16)), int(min(img_h, y1-y0+16))))
            return regs
        except Exception:
            return []
    # merge adjacent hits
    rects=[]
    for r,c in hits:
        x = c*cell; y = r*cell; w = cell; h = cell
        merged=False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                nx=min(rx,x); ny=min(ry,y)
                nw=max(rx+rw, x+w)-nx; nh=max(ry+rh, y+h)-ny
                rects[i]=(nx,ny,nw,nh); merged=True; break
        if not merged:
            rects.append((x,y,w,h))
    # expand slightly
    regs=[]
    for (x,y,w,h) in rects:
        nx=max(0,x-8); ny=max(0,y-8); nw=min(img_w-nx,w+16); nh=min(img_h-ny,h+16)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    return regs

def analyze_board_region(img_bgr, region):
    x,y,w,h = region
    crop = img_bgr[y:y+h, x:x+w]
    pts, mr, mb = detect_red_blue_points(crop)
    if not pts:
        return {"total":0, "maxRun":0, "category":"empty", "flattened":[], "runs":[]}
    # cluster by X into columns
    xs = [p[0] for p in pts]
    ids = sorted(range(len(xs)), key=lambda i: xs[i])
    col_groups=[]
    for i in ids:
        xv = xs[i]
        placed=False
        for grp in col_groups:
            gv = [pts[j][0] for j in grp]; ifv = sum(gv)/len(gv)
            if abs(ifv - xv) <= max(8, w//40):
                grp.append(i); placed=True; break
        if not placed:
            col_groups.append([i])
    # build sequences per column top->bottom
    sequences=[]
    for grp in col_groups:
        col_pts = sorted([pts[i] for i in grp], key=lambda t: t[1])
        seq = [t[2] for t in col_pts]
        sequences.append(seq)
    # flatten per plate reading: row-wise
    flattened=[]
    maxlen = max((len(s) for s in sequences), default=0)
    for r in range(maxlen):
        for col in sequences:
            if r < len(col):
                flattened.append(col[r])
    # compute runs
    runs=[]
    if flattened:
        cur={"color":flattened[0],"len":1}
        for k in range(1,len(flattened)):
            if flattened[k]==cur["color"]: cur["len"]+=1
            else: runs.append(cur); cur={"color":flattened[k],"len":1}
        runs.append(cur)
    maxRun = max((r["len"] for r in runs), default=0)
    # categorize
    if maxRun >= 10: cat="super_long"
    elif maxRun >= 8: cat="long"
    elif maxRun >= 4: cat="longish"
    elif maxRun == 1: cat="single"
    else: cat="other"
    # detect if this board has "multi-row 多连/连珠 in 3 successive rows" (heuristic):
    # we check sequences per column whether there are at least 3 adjacent columns each with top-run>=4
    multi_row = False
    try:
        col_run_lengths = []
        for seq in sequences:
            # biggest top-to-bottom run for same color near top:
            top_run = 1
            for i in range(1, len(seq)):
                if seq[i]==seq[i-1]: top_run+=1
                else: break
            col_run_lengths.append(top_run)
        # look for 3 consecutive columns with run>=4
        cons=0
        for rl in col_run_lengths:
            if rl >=4:
                cons +=1
                if cons >= 3:
                    multi_row=True; break
            else:
                cons=0
    except Exception:
        multi_row=False

    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "flattened":flattened, "runs":runs, "multi_row":multi_row}

# ---------- 页面操作：打开 DG 并尽力处理滑块 ----------
def human_like_move(page, start, end, steps=30):
    """ 模拟人类曲线拖动（小随机） """
    sx, sy = start; ex, ey = end
    for i in range(1, steps+1):
        t = i / steps
        # ease
        x = sx + (ex - sx) * (t**0.9) + random.uniform(-2,2)
        y = sy + (ey - sy) * (t**0.9) + random.uniform(-1,1)
        try:
            page.mouse.move(x, y)
        except Exception:
            pass
        time.sleep(random.uniform(0.006, 0.02))

def try_solve_slider(page):
    """
    尝试寻找页面常见的滑动验证元素并以模拟拖拽方式通过。
    返回 True/False
    """
    try:
        # 多策略查找可能的滑块容器或句柄
        sel_candidates = [
            "div[role=slider]", "div.slider", ".slider", ".drag", ".slide-block", "#slider", ".vaptcha_slider",
            "text=/滑动/", "text=/拖动/"
        ]
        # try to find handle via bounding boxes
        for sel in sel_candidates:
            try:
                els = page.locator(sel)
                if els.count() > 0:
                    el = els.first
                    box = el.bounding_box()
                    if not box:
                        continue
                    # compute start & end
                    sx = box["x"] + box["width"]/4
                    sy = box["y"] + box["height"]/2
                    ex = box["x"] + box["width"]*0.95
                    ey = sy
                    page.mouse.move(sx, sy)
                    page.mouse.down()
                    human_like_move(page, (sx,sy), (ex,ey), steps=random.randint(20,40))
                    page.mouse.up()
                    time.sleep(random.uniform(1.2,2.2))
                    # check some success indicator: slider disappears or page changes
                    try:
                        if not el.is_visible(timeout=1500):
                            return True
                    except Exception:
                        # maybe success even if still visible; let caller check loaded page
                        return True
            except Exception:
                continue

        # fallback: try to drag an element visually near bottom-right quarter of viewport to the right (generic)
        vp = page.viewport_size
        if vp:
            sx = vp["width"]*0.12; sy = vp["height"]*0.6
            ex = vp["width"]*0.88; ey = sy
            page.mouse.move(sx,sy)
            page.mouse.down()
            human_like_move(page, (sx,sy), (ex,ey), steps=random.randint(30,60))
            page.mouse.up()
            time.sleep(1.0)
            return True
    except Exception as e:
        log(f"try_solve_slider 出错: {e}")
    return False

def capture_dg_screenshot(play, url, max_attempts=2, timeout=35000):
    """
    尝试打开 DG 链接并进入实盘页：点击 Free / 免费试玩 -> 处理滑动安全条 -> 等待实盘界面
    成功则返回截图 bytes（PNG）；失败返回 None
    """
    browser = None
    try:
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
        context = browser.new_context(viewport={"width":1280, "height":800})
        page = context.new_page()
        log(f"访问 {url}")
        page.goto(url, timeout=timeout)
        time.sleep(1.2 + random.random()*1.2)
        # 1) 点击 Free / 免费试玩
        clicked = False
        for txt in ["Free", "免费试玩", "免费", "Play Free", "试玩", "free"]:
            try:
                loc = page.get_by_text(txt)
                if loc.count() > 0:
                    loc.first.click(timeout=3000)
                    clicked=True
                    log(f"尝试点击文字按钮: {txt}")
                    break
            except Exception:
                pass
        # 1b) try clicking typical buttons/anchors
        if not clicked:
            candidates = ["button", "a", "input[type=button]"]
            for sel in candidates:
                try:
                    els = page.query_selector_all(sel)
                    for el in els:
                        try:
                            txt = (el.inner_text() or "").strip()
                            if txt and any(k in txt for k in ["Free","免费","试玩","Start","Play"]):
                                el.click()
                                clicked=True
                                log(f"通过元素点击进入（{sel}）: {txt}")
                                break
                        except Exception:
                            continue
                    if clicked: break
                except Exception:
                    continue
        time.sleep(1.0 + random.random()*1.5)
        # 2) 处理滑动安全条（多次尝试）
        success_slider = False
        for attempt in range(3):
            success_slider = try_solve_slider(page)
            log(f"滑块尝试 {attempt+1} -> {success_slider}")
            time.sleep(1.0 + random.random()*1.5)
            # after attempt, check if page content changed to show game area
            try:
                # heuristic: look for keywords or many red/blue dots area
                # wait a bit for game area to render
                time.sleep(2.0)
                # take a small screenshot and see if it contains many colored points
                tmp = page.screenshot()
                pil = pil_from_bytes(tmp); bgr = cv_from_pil(pil)
                pts,_,_ = detect_red_blue_points(bgr)
                if len(pts) > 8:
                    log("滑块后检测到较多点，认为已进入实盘画面。")
                    success_slider = True
                    break
            except Exception as e:
                log(f"滑块后检测异常: {e}")
        # if not success after attempts, still capture page for debug and return None
        if not success_slider:
            log("滑块可能未通过（或页面未加载出实盘），返回截图供调试。")
            try:
                shot = page.screenshot(full_page=True)
                return shot
            except Exception:
                return None
        # 若成功：截图整页
        shot = page.screenshot(full_page=True)
        log("已截取实盘页面截图。")
        return shot
    except Exception as e:
        log(f"capture_dg_screenshot 出错: {e}\n{traceback.format_exc()}")
        return None
    finally:
        try:
            if browser: browser.close()
        except:
            pass

# ---------- 判定总体时段（基于板子统计） ----------
def classify_boards(board_stats):
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    # 中等（中上）额外条件：至少 MID_MULTI_ROW_REQ 张桌子具有 multi_row==True 且 >= MID_LONG_REQ 张为长龙/超长龙
    multi_row_count = sum(1 for b in board_stats if b.get('multi_row', False))
    longishCount = sum(1 for b in board_stats if b['category']=='longish')
    totals = [b['total'] for b in board_stats]
    sparse = sum(1 for t in totals if t < 6)
    n = len(board_stats)
    # 放水判定：至少 MIN_BOARDS_FOR_PAW 张为 长龙/超长龙
    if longCount >= MIN_BOARDS_FOR_PAW:
        return "放水时段（提高胜率）", longCount, superCount, multi_row_count
    # 中等胜率（中上）：有 >= MID_MULTI_ROW_REQ 张 multi_row 且 >= MID_LONG_REQ 张长龙/超长龙
    if multi_row_count >= MID_MULTI_ROW_REQ and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount, multi_row_count
    # 收割判断
    if n>0 and sparse >= n*0.6:
        return "胜率调低 / 收割时段", longCount, superCount, multi_row_count
    return "胜率中等（平台收割中等时段）", longCount, superCount, multi_row_count

# ---------- 主流程 ----------
def main():
    log("开始检测周期。")
    state = load_state()
    now_iso = datetime.now(TZ).isoformat()
    # 1) 访问 DG 并截图
    screenshot = None
    with sync_playwright() as p:
        for url in DG_LINKS:
            try:
                shot = capture_dg_screenshot(p, url)
                if shot:
                    screenshot = shot
                    break
            except Exception as e:
                log(f"访问 {url} 发生异常：{e}")
    if not screenshot:
        log("未能获取任何截图，本次退出。")
        save_state(state)
        return
    # convert to cv
    pil = pil_from_bytes(screenshot)
    bgr = cv_from_pil(pil)
    h,w = bgr.shape[:2]
    pts, mr, mb = detect_red_blue_points(bgr)
    log(f"检测到总点数: {len(pts)}")
    if len(pts) < 6:
        # 可能未进入实盘，保存截图并退出
        log("点数过少，可能并未真正进入实盘界面；将截图写入 last_run_summary.json 供调试。")
        debug = {"ts": now_iso, "error": "Few points", "points": len(pts)}
        with open(LAST_SUMMARY, "w", encoding="utf-8") as f: json.dump(debug, f, ensure_ascii=False, indent=2)
        save_state(state)
        return

    regions = cluster_boards(pts, w, h)
    log(f"聚类得到候选桌子数量: {len(regions)}")
    board_stats=[]
    for reg in regions:
        st = analyze_board_region(bgr, reg)
        board_stats.append(st)
    overall, longCount, superCount, multi_row_count = classify_boards(board_stats)
    log(f"判定：{overall} (长龙/超龍數={longCount}/{superCount}, multi_row_count={multi_row_count})")

    # 保存 summary 以便调试
    debug = {"ts": now_iso, "overall": overall, "longCount": longCount, "superCount": superCount, "multi_row_count": multi_row_count, "boards": board_stats[:40]}
    with open(LAST_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)

    # 状态迁移和提醒逻辑
    was_active = state.get("active", False)
    was_kind   = state.get("kind", None)
    is_active_now = overall in ("放水时段（提高勝率）", "中等勝率（中上）", "中等胜率（中上）", "中等勝率（中上）") or overall == "中等胜率（中上）"
    # standardize exact string check
    is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上)","中等胜率（中上）","中等胜率（中上）")
    # above line ensures matching; to be safe, we'll use simpler membership:
    is_active_now = overall in ("放水时段（提高胜率)","放水时段（提高勝率）","放水时段（提高胜率）","中等胜率（中上）","中等胜率（中上)")

    # simpler check:
    is_active_now = overall in ("放水时段（提高胜率）","中等胜率（中上）")

    if is_active_now and not was_active:
        # start new event
        # estimate end time by history average durations
        hist = state.get("history", [])
        dur_est = None
        if hist:
            durations = [h.get("duration_minutes",0) for h in hist if h.get("duration_minutes",0)>0]
            if durations:
                dur_est = int(sum(durations)/len(durations))
        if not dur_est:
            dur_est = 10
        est_end = (datetime.now(TZ) + timedelta(minutes=dur_est)).strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🚨"
        msg = f"{emoji} <b>DG提醒 — {overall}</b>\n偵測時間 (本地): {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n長/超龍桌數: {longCount} / {superCount}\n滿足多排多連桌數: {multi_row_count}\n估計結束時間: {est_end}（約 {dur_est} 分鐘）"
        ok = send_telegram(msg)
        if ok:
            new_state = {"active": True, "kind": overall, "start_time": datetime.now(TZ).isoformat(), "last_seen": datetime.now(TZ).isoformat(), "history": state.get("history", [])}
            save_state(new_state)
        else:
            # if telegram fail, still record start (so repeated runs won't keep spamming)
            new_state = {"active": True, "kind": overall, "start_time": datetime.now(TZ).isoformat(), "last_seen": datetime.now(TZ).isoformat(), "history": state.get("history", [])}
            save_state(new_state)
    elif is_active_now and was_active:
        # update last seen
        state["last_seen"] = datetime.now(TZ).isoformat()
        state["kind"] = overall
        save_state(state)
    elif (not is_active_now) and was_active:
        # event ended
        start = datetime.fromisoformat(state.get("start_time"))
        end = datetime.now(TZ)
        duration = (end - start).total_seconds()/60.0
        duration_min = int(round(duration))
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": state.get("start_time"), "end_time": end.isoformat(), "duration_minutes": duration_min})
        history = history[-120:]
        new_state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history}
        save_state(new_state)
        emoji = "🔔"
        msg = f"{emoji} <b>DG提醒 — {state.get('kind')} 已結束</b>\n開始: {state.get('start_time')}\n結束: {end.isoformat()}\n實際持續: {duration_min} 分鐘"
        send_telegram(msg)
    else:
        # nothing to do
        save_state(state)
        log("目前非放水/中上時段，不發提醒。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"採集流程發生未處理例外: {e}\n{traceback.format_exc()}")
        sys.exit(1)

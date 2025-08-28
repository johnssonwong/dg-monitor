# main.py
# -*- coding: utf-8 -*-
"""
DG 自动监测脚本（用于 GitHub Actions）
- 会尝试打开 DG，点击 Free -> 模拟滑动安全条 -> 进入实盘 -> 截图 -> 图像识别 -> 判定并发送 Telegram
- 规则按用户在聊天窗口定义（长连≥4, 龙≥8, 超龙≥10, 单跳/双跳等；触发放水或中等胜率中上时提醒）
"""

import os, sys, time, json, math, random
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import requests
import numpy as np
from PIL import Image
import cv2

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ----------------- 用户配置（已自动填入） -----------------
# 注意：出于安全考虑，生产环境应当把 token/chat 放到 GitHub Secrets。
TG_BOT_TOKEN_DEFAULT = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID_DEFAULT  = "485427847"

# 允许以环境变量覆盖
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", TG_BOT_TOKEN_DEFAULT)
TELEGRAM_CHAT_ID  = os.environ.get("TG_CHAT_ID", TG_CHAT_ID_DEFAULT)

# DG 链接（已填）
DG_LINKS = [
    "https://dg18.co/wap/",
    "https://dg18.co/"
]

# 判定阈值（可根据识别效果调）
MIN_BOARDS_FOR_PAW = int(os.environ.get("MIN_BOARDS_FOR_PAW", "3"))  # 放水至少满足桌数
MID_LONG_REQ = int(os.environ.get("MID_LONG_REQ", "2"))             # 中等胜率需要多少张长龙
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "10"))    # 冷却（应用在逻辑层）
H_MIN_POINT_AREA = int(os.environ.get("H_MIN_POINT_AREA","8"))      # 点最小面积
H_MAX_EMPTY_RATIO = float(os.environ.get("H_MAX_EMPTY_RATIO","0.6"))# 用于判断收割

# 状态文件
STATE_FILE = "state.json"
SUMMARY_FILE = "last_run_summary.json"

# Malaysia timezone
TZ_OFFSET = 8
TZ = timezone(timedelta(hours=TZ_OFFSET))

# ----------------- 辅助函数 -----------------
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)

# ----------------- Telegram -----------------
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram 未配置，跳过发送。")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=20)
        jr = r.json()
        if jr.get("ok"):
            log("Telegram 已发送。")
            return True
        else:
            log(f"Telegram 返回错误: {jr}")
            return False
    except Exception as e:
        log(f"发送 Telegram 失败: {e}")
        return False

# ----------------- state 管理 -----------------
def load_state():
    if not os.path.exists(STATE_FILE):
        s = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}
        return s
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": []}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ----------------- 图像检测：点/圆检测 & 颜色分类 -----------------
def pil_from_bytes(b):
    return Image.open(BytesIO(b)).convert("RGB")

def bgr_from_pil(p):
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def detect_circles_and_colors(bgr):
    """
    使用 HoughCircles 检测圆形（珠子），然后采样圆心颜色判断 B(庄/red)/P(闲/blue)。
    返回 points 列表：[(x,y,label), ...]
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)
    h, w = gray.shape
    # Hough 参数需要根据分辨率自适应
    dp = 1.2
    minDist = max(6, int(w/100))
    minRadius = max(3, int(min(w,h)/200))
    maxRadius = max(8, int(min(w,h)/45))
    circles = []
    try:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                   param1=50, param2=20,
                                   minRadius=minRadius, maxRadius=maxRadius)
    except Exception as e:
        log(f"HoughCircles error: {e}")
    points = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            # sample color at center and small surrounding area
            xs = max(0, x-2); xe = min(w-1, x+2)
            ys = max(0, y-2); ye = min(h-1, y+2)
            region = bgr[ys:ye+1, xs:xe+1]
            # average color BGR
            avg = region.reshape(-1,3).mean(axis=0)
            b,g,rcol = avg
            # classify: red if r much larger, blue if b much larger
            if rcol > 140 and rcol > b + 40 and rcol > g + 30:
                label = "B"  # Banker / red
            elif b > 120 and b > rcol + 30 and b > g + 20:
                label = "P"  # Player / blue
            else:
                # uncertain -> skip or attempt HSV test
                hsv = cv2.cvtColor(region.astype("uint8"), cv2.COLOR_BGR2HSV)
                hval = hsv[:,:,0].mean()
                if (hval < 10 or hval > 160):
                    label = "B"
                elif 90 < hval < 130:
                    label = "P"
                else:
                    label = "U"
            points.append((int(x), int(y), label))
    else:
        # fallback: color detection by mask (if no circles)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # red masks
        mask1 = cv2.inRange(hsv, np.array([0,80,60]), np.array([10,255,255]))
        mask2 = cv2.inRange(hsv, np.array([160,80,60]), np.array([179,255,255]))
        mask_r = cv2.bitwise_or(mask1, mask2)
        mask_b = cv2.inRange(hsv, np.array([95,60,40]), np.array([140,255,255]))
        # find contours
        for mask, label in [(mask_r,'B'), (mask_b,'P')]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < H_MIN_POINT_AREA: continue
                M = cv2.moments(c)
                if M["m00"]==0: continue
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                points.append((cx,cy,label))
    return points

# ----------------- 将点聚成“桌子”区域（启发式） -----------------
def cluster_points_to_boards(points, img_w, img_h):
    """
    将点聚类为若干 region (x,y,w,h)
    使用粗网格统计，并合并高密度 cell。
    """
    if not points:
        return []
    cell = max(60, int(min(img_w,img_h)/12))
    cols = math.ceil(img_w/cell); rows = math.ceil(img_h/cell)
    counts = [[0]*cols for _ in range(rows)]
    for (x,y,_) in points:
        cx = min(cols-1, x//cell)
        cy = min(rows-1, y//cell)
        counts[cy][cx] += 1
    thr = 5
    hits = []
    for r in range(rows):
        for c in range(cols):
            if counts[r][c] >= thr:
                hits.append((r,c))
    rects = []
    for (r,c) in hits:
        x = c*cell; y = r*cell; w = cell; h = cell
        merged = False
        for i,(rx,ry,rw,rh) in enumerate(rects):
            if not (x > rx+rw+cell or x+w < rx-cell or y > ry+rh+cell or y+h < ry-cell):
                nx = min(rx,x)
                ny = min(ry,y)
                nw = max(rx+rw, x+w) - nx
                nh = max(ry+rh, y+h) - ny
                rects[i] = (nx,ny,nw,nh)
                merged = True
                break
        if not merged:
            rects.append((x,y,w,h))
    # expand and clip
    regs = []
    for (x,y,w,h) in rects:
        nx = max(0,x-8); ny = max(0,y-8)
        nw = min(img_w - nx, w+16); nh = min(img_h - ny, h+16)
        regs.append((int(nx),int(ny),int(nw),int(nh)))
    # if no rects found, fallback to whole image region
    if not regs:
        regs = [(0,0,img_w,img_h)]
    return regs

# ----------------- 对单个 board region 分析（读列 -> 展平 -> runs） -----------------
def analyze_region(bgr, region):
    x,y,w,h = region
    crop = bgr[y:y+h, x:x+w]
    points = detect_circles_and_colors(crop)
    # transform to local coords
    pts_local = [(px,py,label) for (px,py,label) in points if label in ('B','P')]
    if not pts_local:
        return {"total":0,"maxRun":0,"category":"empty","flattened":[],"runs":[]}
    # cluster by x into columns
    pts_local_sorted = sorted(pts_local, key=lambda t: t[0])
    xs = [p[0] for p in pts_local_sorted]
    # 1D cluster
    clusters = []
    for i,p in enumerate(pts_local_sorted):
        if not clusters:
            clusters.append([p])
        else:
            # compare with last cluster mean x
            meanx = sum([q[0] for q in clusters[-1]]) / len(clusters[-1])
            if abs(p[0] - meanx) <= max(8, w//40):
                clusters[-1].append(p)
            else:
                clusters.append([p])
    sequences = []
    for col in clusters:
        col_sorted = sorted(col, key=lambda t: t[1])  # top->bottom
        seq = [c[2] for c in col_sorted]
        sequences.append(seq)
    # flatten read: column by column, top->bottom
    flattened = []
    maxlen = max([len(s) for s in sequences]) if sequences else 0
    for r in range(maxlen):
        for c in range(len(sequences)):
            if r < len(sequences[c]):
                flattened.append(sequences[c][r])
    # compute runs
    runs = []
    if flattened:
        cur = {"color":flattened[0], "len":1}
        for i in range(1,len(flattened)):
            if flattened[i] == cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur); cur = {"color":flattened[i], "len":1}
        runs.append(cur)
    maxRun = max([r["len"] for r in runs]) if runs else 0
    cat = "other"
    if maxRun >= 10: cat = "super_long"
    elif maxRun >= 8: cat = "long"
    elif maxRun >= 4: cat = "longish"
    elif maxRun == 1: cat = "single"
    # compute long-run count (>=4)
    long_runs_count = sum(1 for r in runs if r["len"]>=4)
    return {"total":len(flattened), "maxRun":maxRun, "category":cat, "flattened":flattened, "runs":runs, "long_runs_count": long_runs_count}

# ----------------- overall classification logic（按用户要求尽力实现） -----------------
def classify_overall(board_stats):
    # board_stats: list of region dicts
    longCount = sum(1 for b in board_stats if b['category'] in ('long','super_long'))
    superCount = sum(1 for b in board_stats if b['category']=='super_long')
    # extra criterion: boards that have >=3 long runs (多连/连珠 across multiple rows)
    boards_with_3_long_runs = sum(1 for b in board_stats if b.get('long_runs_count',0) >= 3)
    # boards with >=2 long runs etc
    longishCount = sum(1 for b in board_stats if b['category']=='longish')
    total_boards = max(1, len(board_stats))
    sparse_boards = sum(1 for b in board_stats if b['total'] < 6)
    # 放水时段判定（尽力）
    # 条件 A: 至少 MIN_BOARDS_FOR_PAW 张桌子属于 long 或 super_long
    # OR 条件 B: 出现 1 个超长龙 + 至少 2 个长龙
    if (longCount >= MIN_BOARDS_FOR_PAW) or (superCount >= 1 and longCount >= 2):
        return "放水时段（提高胜率）", longCount, superCount, boards_with_3_long_runs
    # 中等胜率（中上）：若满足用户要求：有 >=3 张桌子连续出现多连（这里用 boards_with_3_long_runs >= 3）
    # 且至少有 MID_LONG_REQ 张桌子为 龙 或 超龙（可与多连同桌）
    if boards_with_3_long_runs >= 3 and longCount >= MID_LONG_REQ:
        return "中等胜率（中上）", longCount, superCount, boards_with_3_long_runs
    # 若大部分桌子空荡 -> 收割
    if sparse_boards >= total_boards * H_MAX_EMPTY_RATIO:
        return "胜率调低 / 收割时段", longCount, superCount, boards_with_3_long_runs
    return "胜率中等（平台收割中等时段）", longCount, superCount, boards_with_3_long_runs

# ----------------- 尝试进入 DG 页面并截图（包含点击 Free、滑动安全条） -----------------
def capture_dg_with_playwright(play, url, wait_for_secs=6):
    browser = None
    try:
        browser = play.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage"])
        context = browser.new_context(viewport={"width":1280,"height":800}, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36")
        page = context.new_page()
        page.set_default_timeout(30000)
        log(f"打开: {url}")
        page.goto(url)
        time.sleep(2.2 + random.random()*1.5)
        # try to click any button that likely enters the demo: look for multiple possible strings
        enter_texts = ["Free", "免费试玩", "免费", "Play Free", "Try Free", "试玩", "进入"]
        clicked = False
        for txt in enter_texts:
            try:
                locator = page.get_by_text(txt)
                if locator.count()>0:
                    try:
                        locator.first.click(timeout=4000)
                        clicked = True
                        log(f"点击文本: {txt}")
                        break
                    except Exception:
                        pass
            except Exception:
                pass
        # try click known button selectors (common patterns)
        if not clicked:
            selectors = ["button.free", "a.free", ".btn-free", ".enter-button", "button.btn", "a.btn"]
            for sel in selectors:
                try:
                    el = page.query_selector(sel)
                    if el:
                        el.click(timeout=3000)
                        clicked = True
                        log(f"点击 selector: {sel}")
                        break
                except Exception:
                    pass

        time.sleep(1.5)
        # if there is a slider or drag-to-verify container, attempt to find and drag it
        # common patterns: input[type=range], .slider, .drag, .verify-slider, .nc-slider
        slider_selectors = [
            "input[type='range']", ".slider", ".drag", ".verify-slider", ".nc-slider", ".slider-button", ".slide-verify",
            "div[aria-label*='slider']", "div[id*='slider']"
        ]
        dragged = False
        for sel in slider_selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    bbox = el.bounding_box()
                    if bbox:
                        sx = bbox["x"] + 2; sy = bbox["y"] + bbox["height"]/2
                        ex = bbox["x"] + bbox["width"] - 6
                        page.mouse.move(sx, sy)
                        page.mouse.down()
                        # perform a human-like drag with small pauses
                        steps = max(8, int((ex - sx)/6))
                        for i in range(steps):
                            nx = sx + (ex - sx) * (i+1)/steps + random.uniform(-2,2)
                            page.mouse.move(nx, sy + random.uniform(-2,2))
                            time.sleep(0.06 + random.random()*0.02)
                        page.mouse.up()
                        log(f"对 {sel} 执行拖动以完成安全条（尝试）。")
                        dragged = True
                        break
            except Exception:
                continue
        # fallback: try to drag an element that looks like a small circle inside slider container area
        if not dragged:
            try:
                # search for elements with role="slider"
                els = page.query_selector_all("[role='slider']")
                if els:
                    el = els[0]; bbox = el.bounding_box()
                    if bbox:
                        sx = bbox["x"] + 2; sy = bbox["y"] + bbox["height"]/2
                        ex = sx + 220
                        page.mouse.move(sx, sy); page.mouse.down()
                        page.mouse.move(ex, sy, steps=20); page.mouse.up()
                        log("尝试 role=slider 拖动。")
                        dragged = True
            except Exception:
                pass

        # wait some time for page to redirect / load real content
        time.sleep(wait_for_secs + random.random()*2)
        # do a few full-page scrolls to ensure content loads
        try:
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior:'smooth'})")
            time.sleep(0.8)
            page.evaluate("window.scrollTo({top: 0, behavior:'smooth'})")
            time.sleep(0.8)
        except:
            pass
        # final screenshot
        img_bytes = page.screenshot(full_page=True)
        log("截图完成。")
        try:
            context.close()
        except:
            pass
        return img_bytes
    except Exception as e:
        log(f"capture error: {e}")
        return None
    finally:
        try:
            if browser:
                browser.close()
        except:
            pass

# ----------------- 主流程 -----------------
def main():
    log("开始检测循环。")
    state = load_state()
    # 依次尝试两个 DG 链接，直到获得截图
    screenshot = None
    with sync_playwright() as play:
        for url in DG_LINKS:
            try:
                screenshot = capture_dg_with_playwright(play, url, wait_for_secs=5)
                if screenshot:
                    break
            except Exception as e:
                log(f"访问 {url} 失败: {e}")
                continue
    if not screenshot:
        log("未能取得有效截图，结束本次运行并保存 state。")
        save_state(state)
        return

    pil = Image.open(BytesIO(screenshot)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h,w = bgr.shape[:2]
    points = detect_circles_and_colors(bgr)
    log(f"检测到点数量: {len(points)}")
    if not points:
        log("未检测到明显牌点，可能页面未完全进入实盘或布局不匹配。保存快照并结束。")
        # 保存 summary
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({"ts":now_str(),"points":0}, f, ensure_ascii=False, indent=2)
        save_state(state)
        return

    regions = cluster_points_to_boards(points, w, h)
    log(f"聚类得到候选桌子区域数量: {len(regions)}")
    board_stats = []
    for idx, reg in enumerate(regions):
        st = analyze_region(bgr, reg)
        st["region_idx"] = idx+1
        st["region_box"] = reg
        board_stats.append(st)
    overall, longCount, superCount, boardsWith3LongRuns = classify_overall(board_stats)
    log(f"判定: {overall}  (长龙/超长龙={longCount}/{superCount} ; 满足3排多连的桌数={boardsWith3LongRuns})")

    # 保存运行 summary（便于调参）
    summary = {"ts": now_str(), "overall": overall, "longCount": longCount, "superCount": superCount, "boards": board_stats[:40]}
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 状态转换逻辑
    was_active = state.get("active", False)
    was_kind = state.get("kind", None)
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上)".replace(")","")) or overall=="中等胜率（中上）"
    # The messy replacement is defensive to ensure exact match
    is_active_now = overall in ("放水时段（提高胜率）", "中等胜率（中上）")

    now_iso = datetime.now(TZ).isoformat()
    # 如果现在激活 且 之前未激活 -> 发送开始提醒（估算结束时间基于历史）
    if is_active_now and not was_active:
        # new event
        history = state.get("history", [])
        est_minutes = None
        durations = [h.get("duration_minutes",0) for h in history if h.get("duration_minutes",0)>0]
        if durations:
            est_minutes = round(sum(durations)/len(durations))
        else:
            est_minutes = 10  # fallback
        est_end_dt = datetime.now(TZ) + timedelta(minutes=est_minutes)
        est_end_str = est_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🔔"
        msg = (f"{emoji} [DG提醒] {overall} 開始\n偵測時間 (MYT UTC+8): {now_iso}\n"
               f"長/超长龙桌數={longCount}，超长龙={superCount}\n估計結束時間（基於歷史/預估）: {est_end_str}（約 {est_minutes} 分鐘）\n")
        send_telegram(msg)
        # update state
        state = {"active": True, "kind": overall, "start_time": now_iso, "last_seen": now_iso, "history": state.get("history", [])}
        save_state(state)
        log("已記錄並發送開始通知。")
    elif is_active_now and was_active:
        # still active -> update last seen and do nothing else
        state["last_seen"] = now_iso
        state["kind"] = overall
        save_state(state)
        log("仍在活動中，已更新 last_seen。")
    elif (not is_active_now) and was_active:
        # event ended -> compute duration
        start_iso = state.get("start_time")
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.now(TZ)
        duration_min = round((end_dt - start_dt).total_seconds() / 60.0)
        history = state.get("history", [])
        history.append({"kind": state.get("kind"), "start_time": start_iso, "end_time": end_dt.isoformat(), "duration_minutes": duration_min})
        history = history[-100:]
        new_state = {"active": False, "kind": None, "start_time": None, "last_seen": None, "history": history}
        save_state(new_state)
        emoji = "✅"
        msg = (f"{emoji} [DG提醒] {state.get('kind')} 已結束\n開始: {start_iso}\n結束: {end_dt.isoformat()}\n實際持續: {duration_min} 分鐘")
        send_telegram(msg)
        log("事件已結束並發送結束通知。")
    else:
        # not active, do nothing
        save_state(state)
        log("目前不在放水/中上時段，不發提醒。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"主程式發生未處理異常: {e}")
        raise

# scripts/dg_detector.py
import os
import time
import json
import math
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from PIL import Image
import io
import numpy as np
import cv2

# Playwright synchronous API
from playwright.sync_api import sync_playwright

# ------------- 配置参数（可修改） -------------
# 这些参数已按照你的定义：
LONG_LIAN = 4        # 连续>=4 粒 = 长连
CHANG_LONG = 8       # 连续>=8 粒 = 长龙
SUPER_CHANG = 10     # 连续>=10 粒 = 超长龙
DOUBLE_JUMP_MAX = 3  # 2~3 粒 = 双跳
# 整个页面判定阈值（与你设定一致）
MIN_TABLES_FOR_PERCENT = 0.50  # >=50% 符合长连/长龙 视为放水

# ------------- 环境变量（来自 GitHub Secrets） -------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DG_URLS = os.getenv("DG_URLS", "https://dg18.co/ https://dg18.co/wap/").split()
# 用于 commit state.json 的 git user
GIT_USER_NAME = "dg-detector[bot]"
GIT_USER_EMAIL = "dg-detector-bot@example.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "state.json"

# ------------- 工具函数 -------------
def send_telegram_message(text, image_bytes=None, filename="dg_snapshot.png"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat not set. Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # send text first, then send photo (so text always delivered)
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})
    ok = resp.ok
    if image_bytes:
        files = {"photo": (filename, image_bytes)}
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        resp2 = requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": filename}, files=files)
        ok = ok and resp2.ok
    print("Telegram send:", ok)
    return ok

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except:
            pass
    # default
    return {"in_run": False, "run_type": None, "start_ts": None}

def save_state_and_commit(state):
    # write state file
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    # commit the change back to repo using provided GITHUB_TOKEN (the workflow supplies it)
    try:
        subprocess.run(["git", "config", "user.email", GIT_USER_EMAIL], check=True)
        subprocess.run(["git", "config", "user.name", GIT_USER_NAME], check=True)
        subprocess.run(["git", "add", str(STATE_FILE)], check=True)
        subprocess.run(["git", "commit", "-m", f"Update detector state: {state}"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("State saved and committed.")
    except Exception as e:
        print("Commit failed:", e)

# ------------- 图像处理：简单颜色检测 + 连续计数（启发式） -------------
# 因为不同平台图形细节会不同，下面使用颜色阈值（BGR/HSV）检测红/蓝圆点，再聚类为表格区域。
def analyze_image_for_tables(img_bytes):
    """
    输入：整页截图 bytes
    输出：{
      'tables': [ { 'bbox':(x,y,w,h), 'runs': [list of columns runs info], 'max_run_len': int, 'type_flags': {...} }, ... ],
      'summary': {...}
    }
    说明：此函数尽量泛化对“白色框内红色/蓝色圆圈”的检测，返回每桌最大连续长度等数值
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # convert to HSV for stable color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red and blue masks (broad)
    # red has two ranges in HSV
    lower_red1 = np.array([0, 80, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 50]); upper_red2 = np.array([179, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    # blue
    lower_blue = np.array([90, 60, 40]); upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # find red/blue contours (these correspond roughly to circles)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mr = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mb = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

    # combine to show where markers are
    both = cv2.bitwise_or(mr, mb)
    # find connected components to cluster candidate regions (likely table areas)
    num_labels, labels_im = cv2.connectedComponents(both)
    regions = []
    for lab in range(1, num_labels):
        mask = (labels_im == lab).astype("uint8") * 255
        ys, xs = np.where(mask)
        if len(xs) < 30 or len(ys) < 30:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # filter very large (maybe entire page) or tiny
        wbox, hbox = x1-x0, y1-y0
        if wbox < 50 or hbox < 50:
            continue
        if wbox*hbox > 0.9*w*h:  # skip almost-full image
            continue
        regions.append((x0,y0,wbox,hbox))
    # if no regions found, fallback to detect by grid-like layout: try to split page into likely table boxes
    if not regions:
        # fallback: split into a grid of 4x4 upper-left region scanning
        grid_boxes = []
        rows = 4
        cols = 3
        ph = h // rows
        pw = w // cols
        for r in range(rows):
            for c in range(cols):
                grid_boxes.append((c*pw, r*ph, pw, ph))
        regions = grid_boxes

    # For each region, count red/blue markers and estimate run-length along columns
    tables = []
    for (x,y,wb,hb) in regions:
        sub = img[y:y+hb, x:x+wb]
        hsv_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mr_sub = cv2.inRange(hsv_sub, lower_red1, upper_red1) | cv2.inRange(hsv_sub, lower_red2, upper_red2)
        mb_sub = cv2.inRange(hsv_sub, lower_blue, upper_blue)
        # detect centroids of markers
        cnts_r = cv2.findContours(mr_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts_b = cv2.findContours(mb_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        pts = []
        for c in cnts_r:
            (cx,cy,wc,hc) = cv2.boundingRect(c)
            if wc*hc < 10: continue
            pts.append((cx+wc//2, cy+hc//2, 'B'))  # B=Banker(red)
        for c in cnts_b:
            (cx,cy,wc,hc) = cv2.boundingRect(c)
            if wc*hc < 10: continue
            pts.append((cx+wc//2, cy+hc//2, 'P'))  # P=Player(blue)
        if not pts:
            # no markers in region => likely empty table
            tables.append({'bbox':(x,y,wb,hb), 'marker_count':0, 'max_run_len':0, 'runs':[], 'type_flags':{}})
            continue
        # Cluster markers by approximate column (x coordinate) to reconstruct columns top-down
        pts_sorted = sorted(pts, key=lambda p:(p[0], p[1]))
        # quantize columns by x position
        xs = [p[0] for p in pts_sorted]
        if len(xs) == 0:
            tables.append({'bbox':(x,y,wb,hb), 'marker_count':0, 'max_run_len':0, 'runs':[], 'type_flags':{}})
            continue
        # cluster xs into columns
        col_thresh = max(10, wb//20)
        columns = []
        for px,py,pc in pts_sorted:
            placed=False
            for col in columns:
                if abs(col['x'] - px) <= col_thresh:
                    col['pts'].append((px,py,pc))
                    placed=True
                    break
            if not placed:
                columns.append({'x':px, 'pts':[(px,py,pc)]})
        # for each column, sort by y top->bottom and create run string
        runs = []
        max_run_len = 0
        for col in columns:
            col['pts'].sort(key=lambda t:t[1])  # top->bottom
            # generate simplified sequence by collapsing vertically near duplicates
            seq = []
            last_y = None
            for px,py,pc in col['pts']:
                if last_y is None or abs(py-last_y) > 6:
                    seq.append(pc)
                    last_y = py
            runs.append(seq)
            # compute max consecutive in this column
            cur = seq[0] if seq else None
            cur_len = 1 if seq else 0
            local_max = 0
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    cur_len += 1
                else:
                    local_max = max(local_max, cur_len)
                    cur_len = 1
            local_max = max(local_max, cur_len)
            max_run_len = max(max_run_len, local_max)
        # determine flags per user definitions
        type_flags = {}
        type_flags['long_lian'] = max_run_len >= LONG_LIAN
        type_flags['chang_long'] = max_run_len >= CHANG_LONG
        type_flags['super_chang'] = max_run_len >= SUPER_CHANG
        tables.append({
            'bbox':(x,y,wb,hb),
            'marker_count':len(pts),
            'max_run_len':int(max_run_len),
            'runs':runs,
            'type_flags':type_flags
        })
    # produce summary counts
    total_tables = len(tables)
    n_long = sum(1 for t in tables if t['type_flags'].get('long_lian'))
    n_chang = sum(1 for t in tables if t['type_flags'].get('chang_long'))
    n_super = sum(1 for t in tables if t['type_flags'].get('super_chang'))
    summary = {'total_tables': total_tables, 'n_long': n_long, 'n_chang': n_chang, 'n_super': n_super}
    return {'tables': tables, 'summary': summary}

# ------------- 判定逻辑（你给的规则完全实现） -------------
def classify_scene(analysis):
    summ = analysis['summary']
    total = summ['total_tables'] if summ['total_tables']>0 else 1
    pct_long = summ['n_long'] / total
    # Rule 1: 放水（胜率调高）
    # - 满桌长连/长龙类型：若 >= 50% 桌子为长连/长龙 => 放水
    # - 或者 超长龙 + 另外至少 2 张为长龙 => 放水
    is_full_long = pct_long >= MIN_TABLES_FOR_PERCENT
    is_super_combo = (summ['n_super'] >= 1 and summ['n_chang'] >= 2)
    if is_full_long or is_super_combo:
        return ('放水', {'pct_long':pct_long, 'n_chang':summ['n_chang'], 'n_super':summ['n_super']})
    # Rule 2: 中等勝率（中上）
    # 定义：放水特征占比不足 50% 但混合出现（例如有 >=2 桌长龙，且不少长连）
    if summ['n_chang'] >= 2 or (pct_long >= 0.30 and summ['n_chang'] >= 1):
        return ('中等胜率（中上）', {'pct_long':pct_long, 'n_chang':summ['n_chang'], 'n_super':summ['n_super']})
    # Rule 3: 胜率中等（收割中等）
    # 大量单跳、图面空荡、连少
    # Here we use heuristics: if many tables have marker_count small and max_run_len < LONG_LIAN
    tables = analysis['tables']
    empty_tables = sum(1 for t in tables if t['marker_count'] < 6 or t['max_run_len'] < LONG_LIAN)
    pct_empty = empty_tables / total
    if pct_empty >= 0.6:
        return ('胜率中等', {'pct_empty':pct_empty})
    # Rule 4: 胜率调低（收割时段）
    # if almost none have long runs
    if summ['n_chang'] < 1 and pct_long < 0.15:
        return ('收割时段', {'pct_long':pct_long})
    # default fallback
    return ('胜率中等', {'pct_long':pct_long, 'n_chang':summ['n_chang'], 'n_super':summ['n_super']})

# ------------- 放水时长估算（启发式） -------------
def estimate_remaining_minutes(analysis, scene_type):
    # 由于无法精确预测，我们用以下启发式估算：
    # - 找到当前所有 max_run_len，并取平均（代表平台本次“倾向”的连长）
    # - 若有超长龙，假设还会持续 avg_len - current_len 轮（若 >0），并假设 1 轮 ~ 1 分钟（实际根据你观察可调）
    tables = analysis['tables']
    lens = [t['max_run_len'] for t in tables if t['max_run_len']>0]
    if not lens:
        return 0, "无法估算（标记：无连珠样本）"
    avg_len = sum(lens)/len(lens)
    # 当前最长列
    cur_max = max(lens)
    # 估算剩余回合数
    remaining_rounds = max(0, int(round(avg_len - cur_max)))
    # 假设每局约 1 分钟（这是近似；各赌场出牌间隔不同；你可以调整）
    est_minutes = remaining_rounds * 1
    # minimal fallback：若检测到超长龙/长龙，至少给 5 分钟
    if scene_type == '放水' and est_minutes < 5:
        est_minutes = 5
    return est_minutes, f"估算基于当前桌面平均连长={avg_len:.1f}, 当前最大={cur_max}, 估算剩余轮数={remaining_rounds}"

# ------------- 主流程 -------------
def run_detector_once():
    # 1) 访问 DG 页面并截图
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1400,"height":900})
        page = context.new_page()

        screenshot_bytes = None
        for url in DG_URLS:
            try:
                print("尝试打开", url)
                page.goto(url, timeout=30000)
                # 等待页面稳定
                page.wait_for_timeout(2500)
                # 找“Free”或“免费试玩”按钮并点击（多种尝试策略）
                try:
                    # 找带 Free 文本的按钮
                    btn = page.locator("text=Free, text=免费试玩").first
                    if btn:
                        btn.click(timeout=3000)
                        page.wait_for_timeout(2000)
                except Exception as e:
                    print("未找到Free按键或点击失败：", e)
                # 如果弹出新页面（target=_blank），切换到新页面
                pages = context.pages
                pg = pages[-1]
                try:
                    # 寻找安全滑块（通常为 class 或 id），尝试拖动
                    # 采用泛化策略：寻找 input[type=range] 或可拖拽元素
                    slider = None
                    try:
                        slider = pg.locator("input[type=range]").first
                    except:
                        slider = None
                    if slider and slider.count() > 0:
                        box = slider.bounding_box()
                        if box:
                            x = box['x'] + 2
                            y = box['y'] + box['height']/2
                            pg.mouse.move(x,y)
                            pg.mouse.down()
                            pg.mouse.move(x+box['width']*0.9, y, steps=10)
                            pg.mouse.up()
                            pg.wait_for_timeout(1200)
                except Exception as ee:
                    print("slider try failed:", ee)
                # 等待主要桌面载入（此处尝试等待可能包含“table grid”或大量 canvas）
                pg.wait_for_timeout(2000)
                # 最后截取 fullPage screenshot（若无法 fullPage，则viewport）
                try:
                    screenshot_bytes = pg.screenshot(full_page=True)
                except:
                    screenshot_bytes = pg.screenshot()
                # 如果获得截图，则退出循环
                if screenshot_bytes:
                    print("截图成功，长度：", len(screenshot_bytes))
                    break
            except Exception as e:
                print("打开 url 失败：", e)
        browser.close()

    if not screenshot_bytes:
        raise RuntimeError("无法从 DG 获取截图。请确认链接可访问或 Free 流程是否有变化。")

    # 2) 分析截图
    analysis = analyze_image_for_tables(screenshot_bytes)
    scene_type, details = classify_scene(analysis)
    est_min, est_reason = estimate_remaining_minutes(analysis, scene_type)

    # 3) 根据 state.json 决定是否发送通知（仅在放水或中等(中上) 时发送）
    state = load_state()
    now_ts = datetime.now(timezone.utc).timestamp()
    # If scene_type is 放水 or 中等胜率（中上） => should notify (放水强提醒；中上小提醒)
    notify_types = ['放水', '中等胜率（中上）']
    send_notice = False
    notice_type = None
    if scene_type in notify_types:
        if not state.get('in_run'):
            # start new run
            send_notice = True
            notice_type = 'start'
            state['in_run'] = True
            state['run_type'] = scene_type
            state['start_ts'] = now_ts
            state['last_scene'] = scene_type
        else:
            # already in run — only send start once; otherwise skip repeated notifications
            # But if previously run_type different and now stronger (比如从 中等->放水), send upgrade notice
            if state.get('run_type') != scene_type:
                send_notice = True
                notice_type = 'upgrade'
                state['run_type'] = scene_type
                state['last_scene'] = scene_type
    else:
        # current scene NOT a notify type
        if state.get('in_run'):
            # Previously was in run -> now ended
            send_notice = True
            notice_type = 'end'
            start_ts = state.get('start_ts')
            state['in_run'] = False
            state['last_scene'] = scene_type
            # compute duration minutes
            dur_minutes = int(round((now_ts - (start_ts or now_ts))/60))
            state['last_run_duration_min'] = dur_minutes
            state['run_type'] = None
            state['start_ts'] = None

    # 4) 建立通知文字
    now_local = datetime.now(tz=timezone.utc).astimezone(tz=timezone(timedelta(hours=8))) # Malaysia +8
    msg_lines = []
    msg_lines.append(f"📊 <b>DG 局势检测</b> （{now_local.strftime('%Y-%m-%d %H:%M:%S')} 马来西亚时间）")
    msg_lines.append(f"检测结果：<b>{scene_type}</b>")
    if isinstance(details, dict):
        msg_lines.append("详情：" + ", ".join([f"{k}={v}" for k,v in details.items()]))
    msg_lines.append(f"检测桌数： {analysis['summary']['total_tables']}, 长连桌数：{analysis['summary']['n_long']}, 长龙：{analysis['summary']['n_chang']}, 超龙：{analysis['summary']['n_super']}")
    if scene_type in notify_types:
        msg_lines.append(f"提醒级别：{'必须提醒（放水）' if scene_type=='放水' else '小提醒（中等胜率 中上）'}")
        if est_min>0:
            est_end = now_local + timedelta(minutes=est_min)
            msg_lines.append(f"估计放水/持续剩余：{est_min} 分钟，预计结束时间：{est_end.strftime('%H:%M:%S')}（估算）")
        else:
            msg_lines.append("估计放水剩余：无法精确估算（样本不足）")
    if notice_type=='start':
        msg_lines.insert(0, "🔔 <b>放水/中上时段 已检测到（开始）</b>")
    elif notice_type=='upgrade':
        msg_lines.insert(0, "🔺 <b>时段升级通知</b>")
    elif notice_type=='end':
        dur = state.get('last_run_duration_min', 0)
        msg_lines.insert(0, f"✅ <b>放水已结束</b>，共持续 {dur} 分钟（实测）")

    text = "\n".join(msg_lines)

    # 5) 如果需要发送提醒或结束通知，则发 Telegram 并附上截图
    if send_notice:
        # send screenshot too
        image_bytes = screenshot_bytes
        send_telegram_message(text, image_bytes=image_bytes)
    else:
        print("未触发通知。当前 scene:", scene_type)

    # 6) 保存 state（并 commit）
    try:
        save_state_and_commit(state)
    except Exception as e:
        print("保存 state 出错：", e)

    # return for logging
    return {"scene":scene_type, "details":details, "est_min":est_min, "est_reason":est_reason}

if __name__ == "__main__":
    try:
        result = run_detector_once()
        print("Detection result:", result)
    except Exception as e:
        print("检测异常：", e)
        # 若发生异常，发送 Telegram 文字告知（不包含截图）
        try:
            send_telegram_message(f"⚠️ DG Detector 执行出错：{e}\n请检查 workflow 日志。")
        except:
            pass
        raise

# monitor.py
# 说明：在 GitHub Actions 上运行，每次抓取 DG 页面截图并做图像判定。
# 环境变量：
#   TG_TOKEN  - Telegram bot token  (推荐通过 GitHub Secrets 设置)
#   TG_CHAT   - Telegram chat id    (推荐通过 GitHub Secrets 设置)
#   DG_URL_1, DG_URL_2 - 可选，已在 workflow 中设置
#
# state.json 会写回到仓库以记录是否当前处于"放水中"及开始时间 (便于计算结束时长)

import os, json, time, math, traceback, io
from datetime import datetime, timezone, timedelta
import requests
from pathlib import Path

# Playwright headless
from playwright.sync_api import sync_playwright

# image libs
import numpy as np
import cv2
from PIL import Image

# ---------- 配置（你可以按需改） -------------
TG_TOKEN = os.environ.get('TG_TOKEN', '')  # 推荐通过 GitHub Secrets 注入
TG_CHAT  = os.environ.get('TG_CHAT', '')
DG_URLS = [ os.environ.get('DG_URL_1', 'https://dg18.co/wap/'),
            os.environ.get('DG_URL_2', 'https://dg18.co/') ]
# 判定阈值
MIN_BOARDS_FOR_PUTTING = int(os.environ.get('MIN_BOARDS', '3'))  # 放水判定：至少3张桌满足长龙条件
MID_LONG_REQ = int(os.environ.get('MID_LONG_COUNT', '2'))      # 中等胜率需要至少2张长龙
COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN', '8'))       # 触发提醒后的冷却时间(分钟)
STATE_FILE = "state.json"
# ----------------------------------------------

# 如果没有在 env 里设置 token/chat，你可以把它直接写在这里（不推荐）
# TG_TOKEN = '8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8'
# TG_CHAT  = '485427847'

# ----------------- 辅助函数 ------------------
def log(msg):
    print(f"[{datetime.now().astimezone()}] {msg}")

def send_telegram(text):
    token = TG_TOKEN
    chat = TG_CHAT
    if not token or not chat:
        log("Telegram token/chat 未配置，跳过发送。")
        return False, "no-token"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat, "text": text})
        if r.status_code == 200:
            log("Telegram 已发送")
            return True, r.json()
        else:
            log(f"Telegram 发送失败: {r.status_code} {r.text}")
            return False, r.text
    except Exception as e:
        log("Telegram 发送异常: " + str(e))
        return False, str(e)

# 读取/写入 state.json（用于记录是否处在放水中、开始时间）
def read_state():
    p = Path(STATE_FILE)
    if not p.exists():
        return {"in_water": False, "start_ts": None, "last_alert_ts": None}
    try:
        return json.loads(p.read_text())
    except:
        return {"in_water": False, "start_ts": None, "last_alert_ts": None}

def write_state(st):
    Path(STATE_FILE).write_text(json.dumps(st))

# 将 state.json commit 回 repo（由 Actions 的 GITHUB_TOKEN 提交）
def commit_state_back():
    try:
        # 简单 git commit push
        os.system('git config user.name "github-actions[bot]"')
        os.system('git config user.email "41898282+github-actions[bot]@users.noreply.github.com"')
        os.system('git add ' + STATE_FILE)
        os.system('git commit -m "update dg monitor state" || echo "no changes"')
        # use provided GITHUB_TOKEN credential (actions/checkout persisted) to push
        os.system('git push origin HEAD:main || git push')
    except Exception as e:
        log("commit state 异常：" + str(e))

# ---------- 图像处理/检测函数 (简化) ------------
# 目标： 对截图中的红/蓝点做 blob 检测，聚类到若干“board regions”，
# 对每个 region 计算 flattened bead sequence (左列到右列，上到下),
# 计算每个 region 的最大连续 run 长度 (maxRun) -> 用于判断长连/长龙/超长龙。

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_red_blue_centers(cv_img):
    """
    输入 BGR 图像，返回所有检测到的（x,y,color）中心点
    color in {'B' (banker/red), 'P' (player/blue)}
    注意：颜色阈值需要根据实际截图微调
    """
    h, w = cv_img.shape[:2]
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    # 红色（Banker）阈值（HSV） - 可微调
    lower_r1 = np.array([0, 80, 40]); upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([170,80,40]); upper_r2 = np.array([180,255,255])
    mask_r = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)
    # 蓝色（Player）阈值
    lower_b = np.array([90,60,30]); upper_b = np.array([140,255,255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)
    # 清理噪声
    kernel = np.ones((3,3), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kernel, iterations=1)

    # 找到连通域
    centers = []
    for mask, col in [(mask_r, 'B'), (mask_b, 'P')]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 8:   # 过滤过小
                continue
            M = cv2.moments(c)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            centers.append((cx, cy, col))
    return centers

def cluster_regions(centers, img_w, img_h):
    """
    将点聚成若干大区域 (boards)。策略：把点划到网格 cell 中，找高密度 cell，
    合并相邻 cell 得到 bounding boxes。
    返回 boxes: list of (x,y,w,h)
    """
    if len(centers) == 0:
        return []
    cell = max(40, img_w // 12)  # 调整
    cols = math.ceil(img_w / cell); rows = math.ceil(img_h / cell)
    grid = [[0]*cols for _ in range(rows)]
    for (x,y,c) in centers:
        cx = min(cols-1, int(x/cell)); ry = min(rows-1, int(y/cell))
        grid[ry][cx] += 1
    thr = 3  # cell 内点数量阈值
    hits = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] >= thr:
                hits.append((r,c))
    # 合并邻近 cell
    boxes = []
    for (r,c) in hits:
        x = c*cell; y = r*cell; w = cell; h = cell
        merged=False
        for b in boxes:
            bx,by,bw,bh = b
            if not (x > bx+bw+cell or x+w < bx-cell or y > by+bh+cell or y+h < by-cell):
                # expand
                nbx = min(bx, x); nby = min(by,y)
                nbw = max(bx+bw, x+w) - nbx; nbh = max(by+bh, y+h) - nby
                b[0]=nbx; b[1]=nby; b[2]=nbw; b[3]=nbh
                merged=True; break
        if not merged:
            boxes.append([x,y,w,h])
    # clip to image
    boxes = [ (max(0,int(x)), max(0,int(y)), min(img_w,int(w)), min(img_h,int(h))) for x,y,w,h in boxes ]
    return boxes

def analyze_board_subimage(cv_sub):
    """对于单个 board subimage，检测点中心并输出 flattened sequence & runs"""
    centers = get_red_blue_centers(cv_sub)
    if not centers:
        return {"total":0,"flattened":[],"runs":[]}
    # cluster by x (columns)
    xs = [c[0] for c in centers]
    xs_sorted = sorted(xs)
    # 做简单分组：当 x 间距 <= col_gap 则属于同列
    col_gap = max(8, cv_sub.shape[1]//30)
    cols = []
    current = [xs_sorted[0]]
    for i in range(1,len(xs_sorted)):
        if xs_sorted[i] - xs_sorted[i-1] <= col_gap:
            current.append(xs_sorted[i])
        else:
            cols.append(current); current=[xs_sorted[i]]
    cols.append(current)
    # 但我们需要每个点的实际 color和y座标 -> 用 centers 中 nearest x to cluster
    col_centers = []
    for col in cols:
        mean_x = sum(col)/len(col)
        items = [c for c in centers if abs(c[0]-mean_x) <= col_gap+1]
        # sort items by y top->bottom
        items_sorted = sorted(items, key=lambda it: it[1])
        col_centers.append(items_sorted)
    # flattened by row: for row in rows: for col in cols: take col[row] if exists
    max_rows = max((len(c) for c in col_centers), default=0)
    flattened = []
    for row in range(max_rows):
        for c in col_centers:
            if row < len(c):
                flattened.append(c[row][2])  # color letter
    # compute runs
    runs=[]
    if flattened:
        cur = {"color":flattened[0], "len":1}
        for i in range(1,len(flattened)):
            if flattened[i] == cur["color"]:
                cur["len"] += 1
            else:
                runs.append(cur)
                cur = {"color":flattened[i], "len":1}
        runs.append(cur)
    return {"total":len(flattened), "flattened":flattened, "runs":runs}

# 判定整体局势
def decide_overall(board_stats):
    longCount = 0
    superCount = 0
    longishCount = 0
    sparse_count = 0
    for b in board_stats:
        runs = b["runs"]
        maxRun = max((r["len"] for r in runs), default=0)
        if maxRun >= 10:
            superCount += 1
        if maxRun >= 8:
            longCount += 1
        elif maxRun >= 4:
            longishCount += 1
        if b["total"] < 6:
            sparse_count += 1
    nboards = max(1, len(board_stats))
    if longCount >= MIN_BOARDS_FOR_PUTTING:
        overall = "放水时段（提高胜率）"
    elif longCount >= MID_LONG_REQ and longishCount > 0:
        overall = "中等胜率（中上）"
    else:
        if sparse_count >= nboards * 0.6:
            overall = "胜率调低 / 收割时段"
        else:
            overall = "胜率中等（平台收割中等时段）"
    return overall, longCount, superCount

# ------------ 主流程 ------------
def run_once():
    log("开始一次检测流程...")
    # Playwright: open browser, try two urls
    shot = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1366, "height":768})
        ok = False
        for url in DG_URLS:
            try:
                log("尝试访问: " + url)
                page.goto(url, timeout=30000)
                time.sleep(2)
                # 尝试点击 "Free" / "免费试玩" 文本
                try:
                    for t in ["Free", "免费试玩", "免费", "START", "试玩"]:
                        btns = page.locator(f'text="{t}"')
                        if btns.count() > 0:
                            try:
                                btns.first.click(timeout=3000)
                                log(f"点击文字: {t}")
                                time.sleep(1.5)
                                break
                            except:
                                pass
                except Exception as e:
                    pass
                # 如果出现一个滑动安全条（常见的 JS 验证），尝试找到滑块并拖动
                try:
                    # 尝试常见滑块元素选择器
                    slider = None
                    selectors = [
                        'div[class*="slider"]', 'input[type="range"]', 'div[id*="slider"]',
                        'div[class*="verification"]', 'div[class*="drag"]'
                    ]
                    for s in selectors:
                        if page.query_selector(s):
                            slider = page.query_selector(s)
                            break
                    if slider:
                        box = slider.bounding_box()
                        if box:
                            sx = box["x"] + 5; sy = box["y"] + box["height"]/2
                            tx = sx + box["width"] - 10
                            page.mouse.move(sx, sy)
                            page.mouse.down()
                            page.mouse.move(tx, sy, steps=15)
                            page.mouse.up()
                            log("尝试模拟滑块拖动")
                            time.sleep(2)
                except Exception as e:
                    log("滑块模拟异常: " + str(e))
                # 等待可能的牌桌区域加载; 以某些已知页面元素做等待
                try:
                    page.wait_for_timeout(2500)
                except:
                    pass
                # 最后截个全页面图
                shot = page.screenshot(full_page=True)
                ok = True
                break
            except Exception as e:
                log("访问或点击失败: " + str(e))
                continue
        browser.close()
    if not shot:
        log("未能获得页面截图，结束本次检测。")
        return None

    # 读取图像
    img_pil = Image.open(io.BytesIO(shot)).convert("RGB")
    cv_img = pil_to_cv(img_pil)
    h,w = cv_img.shape[:2]
    # 检测点中心
    centers = get_red_blue_centers(cv_img)
    if not centers:
        log("未检测到红/蓝点，可能页面未正确进入/截图与界面不匹配。")
    boxes = cluster_regions(centers, w, h)
    if not boxes:
        # 如果 cluster 失败，使用全图分割成若干列作为 fallback
        boxes = [ (0,0,w,h) ]
    board_stats = []
    for (x,y,ww,hh) in boxes:
        sub = cv_img[y:y+hh, x:x+ww]
        st = analyze_board_subimage(sub)
        board_stats.append(st)
    overall, longCount, superCount = decide_overall(board_stats)
    # log some summary
    log(f"检测结果：{overall}  (长/超长龙数: {longCount}, 超长龙数: {superCount}, 检测桌数: {len(board_stats)})")
    return {
        "overall": overall,
        "longCount": longCount,
        "superCount": superCount,
        "nboards": len(board_stats),
        "boards": board_stats
    }

# ---------- 主执行且处理 state 与提醒 ----------
def main():
    global TG_TOKEN, TG_CHAT
    # 允许从脚本内硬编码覆盖（慎用）
    if not TG_TOKEN:
        log("警告: TG_TOKEN 未配置，若要发送 Telegram ，请在 GitHub Secrets 设置 TG_TOKEN")
    if not TG_CHAT:
        log("警告: TG_CHAT 未配置，若要发送 Telegram ，请在 GitHub Secrets 设置 TG_CHAT")
    state = read_state()
    try:
        res = run_once()
        if res is None:
            return
        overall = res["overall"]
        now_ts = int(time.time())
        # 判断是否属于要提醒的两种时段
        is_water_or_mid = (overall == "放水时段（提高胜率）" or overall == "中等胜率（中上）")
        # 如果进入放水且之前未处于放水 -> 发送开始提醒并记录 start_ts
        if is_water_or_mid and not state.get("in_water", False):
            # 检查冷却 last_alert_ts
            last_alert = state.get("last_alert_ts")
            if last_alert and now_ts - last_alert < COOLDOWN_MINUTES*60:
                log("仍在冷却期内，不重复发送放水开始提醒。")
            else:
                text = f"[DG提醒] 現在判定：{overall}\\n長/超长龙={res['longCount']}, 超长龙={res['superCount']}, 檢測桌={res['nboards']}\\n時間：{datetime.now().astimezone().isoformat()}"
                ok, ret = send_telegram(text)
                if ok:
                    state["in_water"] = True
                    state["start_ts"] = now_ts
                    state["last_alert_ts"] = now_ts
                    write_state(state)
                    commit_state_back()
        # 如果当前非放水，但 state 表示之前处于放水 -> 发送放水结束并计算持续时间
        elif (not is_water_or_mid) and state.get("in_water", False):
            start_ts = state.get("start_ts")
            if start_ts:
                duration_min = int((now_ts - start_ts)/60)
                start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).astimezone().isoformat()
                text = f"[DG提醒] 放水結束。\\n開始時間：{start_str}\\n結束時間：{datetime.now().astimezone().isoformat()}\\n共持續約 {duration_min} 分鐘。"
            else:
                text = f"[DG提醒] 放水結束 (時長不明)。"
            send_telegram(text)
            # 清除状态
            state["in_water"] = False
            state["start_ts"] = None
            state["last_alert_ts"] = int(time.time())
            write_state(state)
            commit_state_back()
        else:
            log("当前状态与历史状态一致，或不需要动作。")
    except Exception as e:
        log("主流程异常: " + str(e))
        log(traceback.format_exc())

if __name__ == "__main__":
    main()

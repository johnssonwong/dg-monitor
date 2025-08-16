# run.py
# DG 监控脚本 — Playwright + OpenCV 实现的“放水/中等胜率”自动检测并通过 Telegram 推送
# 已将你的 TG token / chat id / DG 链接内置（请谨慎保管）

import os
import sys
import time
import json
import math
import requests
import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# --------- 配置区（你要的常量，已按照你要求放入） ----------
TG_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
TG_CHAT_ID = "485427847"
DG_URLS = ["https://dg18.co/wap/", "https://dg18.co/"]  # 主站两个链接
TIMEZONE = "Asia/Kuala_Lumpur"  # 只是记录用（你已指定 UTC+8）

# 判定阈值（可按需微调）
DRAGON_LENGTH = 8          # 连续>=8 粒 = 长龙
SUPER_DRAGON_LENGTH = 10   # 连续>=10 粒 = 超长龙
LONG_CHAIN_MIN_FOR_FULL20 = 8   # 当总桌数 >=20，符合放水的桌子至少 >=8
LONG_CHAIN_MIN_FOR_10 = 4       # 当总桌数 >=10，符合放水的桌子至少 >=4

MIDDLE_MIN_FOR_20 = 6  # 中等胜率(中上)：20桌时至少6张符合
MIDDLE_MIN_FOR_10 = 3  # 中等胜率(中上)：10桌时至少3张符合

# 本地工作文件（workflow 会 commit 回 repo）
STATE_FILE = "state.json"

# 其他调试开关
DO_SAVE_DEBUG_SCREENSHOT = False
DEBUG_DIR = "debug"

# ---- Telegram helper ----
def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=15)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def send_telegram_photo(image_bytes, caption=""):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    files = {"photo": ("screenshot.jpg", image_bytes)}
    data = {"chat_id": TG_CHAT_ID, "caption": caption, "parse_mode":"HTML"}
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# ---- 状态持久化 ----
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, "r", encoding="utf-8"))
        except:
            return {}
    return {}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

# ---- 截图 -> 分析：利用 OpenCV 检测红色/蓝色小圆点并估算连珠/长龙 ----
def analyze_image_numpy(np_img):
    """输入：BGR numpy 图像（OpenCV 格式）
       输出：判定结果字典 { 'total_tables_est': N, 'dragon_tables': k1, 'super_dragon_tables': k2, 'middle_candidates': k3, ... }
       实现思路（简化并尽力鲁棒）：
         1) 找到红/蓝颜色掩码
         2) 找到每个小圆形轮廓的质心与颜色
         3) 将这些质心按空间聚类（把一堆圆点分成多个“桌子区域”）
         4) 对每个区域，根据竖向（或行方向）连续相同颜色的数量，估算是否为长龙/超长龙/连珠等
    """
    out = {"total_tables_est": 0, "dragon_tables": 0, "super_dragon_tables": 0, "midchain_tables": 0}
    img = np_img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 颜色阈值（可微调）
    # 红色可能出现在两个Hue区间
    lower_red1 = np.array([0, 80, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 50]); upper_red2 = np.array([179, 255, 255])
    lower_blue = np.array([90, 60, 50]); upper_blue = np.array([140, 255, 255])

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 去噪，扩展
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 找轮廓（小圆点）
    contours_r = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_b = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    points = []  # (x,y,color) color: 'R' or 'B'
    def contours_to_points(contours, color):
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 忽略过小
                continue
            (x,y,w,h) = cv2.boundingRect(cnt)
            cx = int(x + w/2); cy = int(y + h/2)
            points.append((cx, cy, color, area))
    contours_to_points(contours_r, 'R')
    contours_to_points(contours_b, 'B')

    if len(points) == 0:
        return out  # 没检测到任何点，直接返回

    # 聚类为桌子区域（简单聚类，基于靠近原则）
    clusters = []  # 每个 cluster: { 'pts': [...], 'cx':..., 'cy':... }
    for (x,y,c,area) in points:
        placed = False
        for cl in clusters:
            # 若在已有簇中心的水平距离不大（桌子区域通常在横向分布），则加入
            if abs(x - cl['cx']) < 220 and abs(y - cl['cy']) < 200:
                cl['pts'].append((x,y,c))
                # 更新中心
                xs = [p[0] for p in cl['pts']]; ys = [p[1] for p in cl['pts']]
                cl['cx'] = int(sum(xs)/len(xs)); cl['cy'] = int(sum(ys)/len(ys))
                placed = True
                break
        if not placed:
            clusters.append({'pts': [(x,y,c)], 'cx': x, 'cy': y})

    total_tables_est = len(clusters)
    out['total_tables_est'] = total_tables_est

    # 针对每个 cluster，计算竖直方向连续同色“数量”估算长龙
    for cl in clusters:
        pts = cl['pts']
        # 以 x 位置进行列分组（将近似同一竖列视为一列）
        # 先按 x 排序再将近邻合并为列
        pts_sorted = sorted(pts, key=lambda p:(p[0], p[1]))
        # 合并 x 值近的点为列
        columns = []
        for p in pts_sorted:
            x,y,col = p
            if not columns:
                columns.append({'xs':[x], 'pts':[p]})
            else:
                last = columns[-1]
                # 如果 x 与 last 平均 x 距离不大，则归为一列
                last_x = sum(last['xs'])/len(last['xs'])
                if abs(x - last_x) < 25:
                    last['xs'].append(x); last['pts'].append(p)
                else:
                    columns.append({'xs':[x], 'pts':[p]})
        # 在每列里，按 y 排序后计算同色连续 run
        max_run = 0
        any_mid = False
        for col in columns:
            pts_col = sorted(col['pts'], key=lambda t:t[1])
            # 将同色连续统计
            current_color = None
            cur_count = 0
            prev_y = None
            for (x,y,colc) in pts_col:
                if prev_y is None:
                    current_color = colc
                    cur_count = 1
                    prev_y = y
                else:
                    # 若与上一个点垂直距离不大（认为是连续排列），则视为连续
                    if abs(y - prev_y) < 30:
                        if colc == current_color:
                            cur_count += 1
                        else:
                            # color changed -> record run
                            max_run = max(max_run, cur_count)
                            if cur_count >= 4:
                                any_mid = True
                            # reset
                            current_color = colc
                            cur_count = 1
                        prev_y = y
                    else:
                        # 距离太大，断开
                        max_run = max(max_run, cur_count)
                        if cur_count >= 4:
                            any_mid = True
                        current_color = colc
                        cur_count = 1
                        prev_y = y
            # end for points in column
            max_run = max(max_run, cur_count)
            if cur_count >= 4:
                any_mid = True

        # 根据 max_run 判定该 cluster 是否为长龙/超长龙/中等连
        if max_run >= SUPER_DRAGON_LENGTH:
            out['super_dragon_tables'] += 1
        elif max_run >= DRAGON_LENGTH:
            out['dragon_tables'] += 1
        elif max_run >= 4:
            out['midchain_tables'] += 1

    return out

# ---- Playwright automation (screenshot) ----
def take_lobby_screenshot():
    """
    使用 playwright 打开 DG 链接，尝试点击 Free（或“免费试玩”），尝试滑动安全条，
    然后等待大厅出现并截屏整个页面（viewport）。
    返回： bytes of png
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    # 尝试多个 URL
    for url in DG_URLS:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
                context = browser.new_context(viewport={"width":1366, "height":768})
                page = context.new_page()
                page.set_default_timeout(25000)
                page.goto(url)
                time.sleep(1.2)

                # 尝试点击 Free / 免费试玩 的按钮（多语言兼容）
                clicked = False
                try_texts = ["text=Free", "text=Free Play", "text=免费试玩", "text=试玩", "text=Free Play"]
                for t in try_texts:
                    try:
                        el = page.locator(t)
                        if el.count() > 0:
                            el.first.click(timeout=3000)
                            clicked = True
                            time.sleep(1.0)
                            break
                    except Exception:
                        pass

                # 如果页面出现滑动验证（常见的是滑块），尝试通过拖动模拟
                # 尝试一组常见选择器
                slider_selectors = [
                    ".geetest_slider_button",    # geetest
                    ".nc_iconfont.btn_slide",    # nc
                    ".drag", ".slider", "#slider",
                    ".slideBlock", ".verification-slider"
                ]
                for sel in slider_selectors:
                    try:
                        if page.locator(sel).count() > 0:
                            box = page.locator(sel).first.bounding_box()
                            if box:
                                # perform drag
                                start_x = box["x"] + box["width"]/2
                                start_y = box["y"] + box["height"]/2
                                # drag to right
                                page.mouse.move(start_x, start_y)
                                page.mouse.down()
                                page.mouse.move(start_x + box["width"]*6, start_y, steps=20)
                                time.sleep(0.4)
                                page.mouse.up()
                                time.sleep(1.0)
                    except Exception:
                        pass

                # 如果仍未进入，尝试简单滚动页面并等待
                page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.0)

                # 等待大约游戏列表载入（尝试几种常见的容器）
                candidates = ["div.lobby", ".room-list", ".game-list", ".table-list", ".lobby-wrap", "div[class*='room']", "body"]
                for c in candidates:
                    try:
                        el = page.locator(c)
                        if el.count() > 0:
                            # 等待短暂时间以稳定画面
                            time.sleep(1.0)
                            break
                    except Exception:
                        pass

                # 最后对页面做全页面截图
                screenshot_bytes = page.screenshot(full_page=True)
                browser.close()
                return screenshot_bytes
        except Exception as e:
            # 尝试下一个 URL
            print("visit url error", url, e)
            continue
    raise RuntimeError("无法通过 Playwright 获取页面截图，请检查网络或目标站点结构/防护。")

# ---- 判定主流程 ----
def evaluate_and_report():
    send_telegram_text("📡 DG 监控：开始一次检测（UTC+8 时间）")
    start_run = datetime.datetime.utcnow()
    try:
        png_bytes = take_lobby_screenshot()
    except Exception as e:
        send_telegram_text(f"❗ 无法截取 DG 页面：{e}")
        return

    # 保存调试图（如果需要）
    if DO_SAVE_DEBUG_SCREENSHOT:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        open(os.path.join(DEBUG_DIR, "last.png"), "wb").write(png_bytes)

    # 读取为 cv2 图像
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 分析
    result = analyze_image_numpy(img)
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)  # 转成 UTC+8 仅用于显示
    timestr = now.strftime("%Y-%m-%d %H:%M:%S")
    summary = f"检测时间：{timestr} (UTC+8)\n总估计桌数: {result['total_tables_est']}\n超长龙桌: {result['super_dragon_tables']}\n长龙桌: {result['dragon_tables']}\n中等连候选: {result['midchain_tables']}"
    print(summary)

    # 判定是否为 放水 / 中等胜率（中上） / 其他（不提醒）
    total = result['total_tables_est']
    dragons = result['dragon_tables']
    supers = result['super_dragon_tables']
    mids = result['midchain_tables']

    is_push = False
    reason = ""
    mode = None  # "full" / "middle" / None

    # 1) 满盘长连局势型 放水判定（20桌或10桌规则）
    if total >= 20 and (dragons + supers) >= LONG_CHAIN_MIN_FOR_FULL20:
        is_push = True; mode = "放水时段（提高胜率）"; reason = f"20桌≥，符合桌数 {(dragons+supers)} >= {LONG_CHAIN_MIN_FOR_FULL20}。"
    elif total >= 10 and (dragons + supers) >= LONG_CHAIN_MIN_FOR_10:
        is_push = True; mode = "放水时段（提高胜率）"; reason = f"10桌≥，符合桌数 {(dragons+supers)} >= {LONG_CHAIN_MIN_FOR_10}。"
    # 2) 超长龙触发型
    elif supers >= 1 and dragons >= 2 and (supers + dragons) >= 3:
        is_push = True; mode = "放水时段（提高胜率）"; reason = f"存在超长龙与至少2长龙：超 {supers}，长 {dragons}。"
    else:
        # 3) 中等胜率（中上）：介于放水与一般收割之间（符合中上规则）
        if total >= 20 and (dragons + supers + mids) >= MIDDLE_MIN_FOR_20:
            is_push = True; mode = "中等胜率（中上）"; reason = f"20桌≥，符合 {dragons+supers+mids} >= {MIDDLE_MIN_FOR_20}。"
        elif total >= 10 and (dragons + supers + mids) >= MIDDLE_MIN_FOR_10 and (dragons+supers) >= 2:
            # 额外要求：至少 2 桌有长龙/超长龙
            is_push = True; mode = "中等胜率（中上）"; reason = f"10桌≥，{dragons+supers+mids}>= {MIDDLE_MIN_FOR_10}，且长龙≥2。"

    # 读取状态文件（用于记录放水开始/结束）
    state = load_state()
    now_ts = int(time.time())

    if is_push:
        # 如果之前 state 没有运行标记（running），则写入 start
        if not state.get("running"):
            state["running"] = True
            state["start_ts"] = now_ts
            state["mode"] = mode
            save_state(state)
            # 发送开始消息（带截图）
            caption = f"🔔 <b>{mode}</b>\n判定原因：{reason}\n{summary}\n动作：开始提醒（开始时间记录）"
            try:
                send_telegram_photo(png_bytes, caption=caption)
            except Exception:
                send_telegram_text("🔔 放水/中上检测到，但发送图片失败，已发送文本。")
                send_telegram_text(caption)
        else:
            # 已经在放水状态，更新但不重复发送（每次运行可发送一次状态更新，或不发送）
            # 我们这里选择：只在首次进入才发送提醒，后续轮询不发送重复提醒，避免刷屏
            print("状态：已在放水/中上运行中，不再重复提醒。")
    else:
        # 当前检测不为放水。若 state 标记 running=True，则表示放水刚结束 → 计算持续时间并发送结束消息
        if state.get("running"):
            start_ts = state.get("start_ts", now_ts)
            duration_min = int((now_ts - start_ts) / 60)
            # 清除运行状态
            state["running"] = False
            state["last_duration_min"] = duration_min
            state["last_end_ts"] = now_ts
            save_state(state)
            # 发送放水已结束信息
            caption = f"⏹ 放水/中上 已结束\n模式：{state.get('mode')}\n持续时间：{duration_min} 分钟\n检测时间：{timestr}\n判定摘要：{summary}"
            send_telegram_text(caption)
            # 也附带最后一次截图
            try:
                send_telegram_photo(png_bytes, caption=caption)
            except:
                pass
        else:
            print("当前不是放水/中上时段，不提醒。")

    # 最后，无论如何把 state.json 写回（workflow 会 commit）
    save_state(state)

if __name__ == "__main__":
    try:
        evaluate_and_report()
    except Exception as e:
        send_telegram_text(f"❗DG 监控脚本发生异常：{e}")
        raise

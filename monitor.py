# monitor.py
# DreamGaming (DG) 自动监测脚本（供 GitHub Actions 运行）
# 功能：
# - 自动打开 DG（两个可选 URL），尝试点击 "Free"/"免费试玩" 并滚动安全条
# - 截取页面上可能的路单（canvas 或包含路单的 div）
# - 对每个“桌面”截图做颜色/列/竖向连续 run 分析，统计长连/长龙/超长龙/满盘密集度
# - 按你提供的所有判定规则（放水 / 中等胜率（中上） / 胜率中等 / 收割）判断局势
# - 当判定为“放水”或“中等胜率（中上）”时发送 Telegram 警报（并截图）
# - 当放水结束时发送“放水已结束，共持续 XX 分钟”并把本次时长写入历史（用于下次估计）
#
# 请在 GitHub Actions 的 workflow 中通过 Secrets 注入下列环境变量：
# - TELEGRAM_BOT_TOKEN  （你的 bot token）
# - TELEGRAM_CHAT_ID    （你的 chat id）
# - DG_URL1 （可选，默认为 https://dg18.co/wap/）
# - DG_URL2 （可选，默认为 https://dg18.co/）
#
# 注意：脚本尽力自动处理跳出页面与简单滑动，但若遇到复杂真人滑块/反自动化时会失败并报告失败需要人工干预。
# ------------------------------------------------------------------------------

import os, time, json, math, statistics
from datetime import datetime
import requests
import numpy as np
from PIL import Image
import cv2
from playwright.sync_api import sync_playwright

# ========== 配置 & 环境变量 ==========
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
DG_URLS = [
    os.environ.get("DG_URL1", "https://dg18.co/wap/"),
    os.environ.get("DG_URL2", "https://dg18.co/")
]

STATE_FILE = "state.json"

# 判定参数（你可以按需微调）
MIN_LONG = 4            # 长连（≥4）
MIN_LONGCHAIN = 8       # 长龙（≥8）
MIN_SUPERLONG = 10      # 超长龙（≥10）
NONEMPTY_RATIO_FULL = 0.45
COL_PEAK_RATIO = 0.08
IMG_MAXHEIGHT = 360

# ========== Telegram 帮助函数 ==========
def send_telegram_text(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 未配置，跳过发送:", text[:120])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=20)
        print("Telegram sendMessage status:", r.status_code)
    except Exception as e:
        print("Telegram send message failed:", e)

def send_telegram_photo(img_bytes, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 未配置，跳过发图")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("screenshot.jpg", img_bytes)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, files=files, timeout=60)
        print("Telegram sendPhoto", r.status_code)
    except Exception as e:
        print("send photo failed:", e)

# ========== 状态管理 ==========
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"status":"idle","alert_type":None,"start_ts":None,"history_minutes":[]}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def commit_state_git():
    # 在 Actions 中会有权限 push，如果没有权限此命令不会中断
    os.system('git config user.email "actions@github.com"')
    os.system('git config user.name "github-actions"')
    os.system(f"git add {STATE_FILE} || true")
    os.system('git commit -m "update monitor state" || true')
    os.system("git push || true")

# ========== 图像分析 ==========
def analyze_board_image_bytes(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h,w = img.shape[:2]
        if h > IMG_MAXHEIGHT:
            scale = IMG_MAXHEIGHT/h
            img = cv2.resize(img, (int(w*scale), IMG_MAXHEIGHT), interpolation=cv2.INTER_AREA)
            h,w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0,70,50]); upper1 = np.array([10,255,255])
        lower2 = np.array([170,70,50]); upper2 = np.array([180,255,255])
        red1 = cv2.inRange(hsv, lower1, upper1); red2 = cv2.inRange(hsv, lower2, upper2)
        red = cv2.bitwise_or(red1, red2)
        lowerb = np.array([90,70,50]); upperb = np.array([140,255,255])
        blue = cv2.inRange(hsv, lowerb, upperb)

        nonempty = cv2.bitwise_or(red, blue)
        nonempty_ratio = float(np.count_nonzero(nonempty)) / (h*w)

        proj = np.sum(nonempty, axis=0).astype(np.float32)
        if proj.max() <= 0:
            return {"longest_red":0,"longest_blue":0,"nonempty_ratio":nonempty_ratio}

        kernel = np.ones(7)/7.0
        proj_s = np.convolve(proj, kernel, mode='same')
        threshold = proj_s.max() * COL_PEAK_RATIO

        cols = []
        in_seg=False; s=0
        for i,v in enumerate(proj_s):
            if v > threshold and not in_seg:
                in_seg=True; s=i
            if v <= threshold and in_seg:
                in_seg=False; cols.append((s,i))
        if in_seg:
            cols.append((s,len(proj_s)))
        if len(cols)==0:
            return {"longest_red":0,"longest_blue":0,"nonempty_ratio":nonempty_ratio}

        longest_red=0; longest_blue=0
        for (sx,ex) in cols:
            col_red_rows = (np.sum(red[:, sx:ex], axis=1) > 0)
            col_blue_rows = (np.sum(blue[:, sx:ex], axis=1) > 0)
            def max_run(arr):
                maxr=0; cur=0
                for b in arr:
                    if b: cur+=1; maxr=max(maxr,cur)
                    else: cur=0
                return maxr
            rrun = max_run(col_red_rows); brun = max_run(col_blue_rows)
            longest_red = max(longest_red, rrun)
            longest_blue = max(longest_blue, brun)

        return {"longest_red":int(longest_red),"longest_blue":int(longest_blue),"nonempty_ratio":float(nonempty_ratio)}
    except Exception as e:
        print("analyze error", e)
        return None

# ========== 全局判定规则（使用你提供的所有规则） ==========
def classify_tables(table_metrics):
    total = len(table_metrics)
    cnt_full_like = 0
    cnt_long = 0
    cnt_superlong = 0
    for m in table_metrics:
        if m is None: continue
        longest = max(m.get("longest_red",0), m.get("longest_blue",0))
        if m.get("nonempty_ratio",0) >= NONEMPTY_RATIO_FULL and longest >= MIN_LONG:
            cnt_full_like += 1
        if longest >= MIN_LONGCHAIN:
            cnt_long += 1
        if longest >= MIN_SUPERLONG:
            cnt_superlong += 1

    full_judge = False
    if total >= 20 and cnt_full_like >= 8:
        full_judge = True
    if total >= 10 and cnt_full_like >= 4:
        full_judge = True

    super_judge = False
    if cnt_superlong >= 1 and cnt_long >= 3:
        super_judge = True

    mid_high_judge = False
    if total >= 20 and cnt_full_like >= 6: mid_high_judge = True
    if total >= 10 and cnt_full_like >= 3: mid_high_judge = True
    if cnt_long >= 2 and (cnt_full_like > 0): mid_high_judge = True

    sparse_count = sum(1 for m in table_metrics if m and m.get("nonempty_ratio",0) < 0.15)
    low_rate_judge = False
    if total > 0 and sparse_count/total > 0.6 and cnt_long < 2:
        low_rate_judge = True

    result = {
        "total_tables": total,
        "cnt_full_like": cnt_full_like,
        "cnt_long": cnt_long,
        "cnt_superlong": cnt_superlong,
        "full_judge": full_judge,
        "super_judge": super_judge,
        "mid_high_judge": mid_high_judge,
        "low_rate_judge": low_rate_judge
    }
    if full_judge or super_judge:
        result["label"] = "放水"
    elif mid_high_judge:
        result["label"] = "中等胜率(中上)"
    elif low_rate_judge:
        result["label"] = "收割/胜率调低"
    else:
        result["label"] = "胜率中等"
    return result

# ========== Playwright 抓取页面并截图关键区域 ==========
def try_load_dg_and_screenshot(playwright):
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu"])
    context = browser.new_context(viewport={"width":1400,"height":900})
    page = context.new_page()
    success=False
    for url in DG_URLS:
        try:
            page.goto(url, timeout=45000)
            time.sleep(1.2)
            # 尝试点击 free/免费/免费试玩 文本
            found=False
            for txt in ["Free","免费","免费试玩","Free Play"]:
                try:
                    el = page.locator(f"text={txt}")
                    if el.count() > 0:
                        el.first.click(timeout=6000)
                        found=True
                        break
                except Exception:
                    pass
            # 尝试滚动安全条区域
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.0)
            # 等待可能出现 canvas 或路单容器
            try:
                page.wait_for_selector("canvas, .roadmap, .game-table, .bt-table", timeout=12000)
            except Exception:
                pass
            success=True
            break
        except Exception as e:
            print("open url failed:", e)
            continue
    if not success:
        browser.close()
        raise RuntimeError("无法打开 DG 页面或被阻挡")

    board_images=[]
    try:
        canvases = page.query_selector_all("canvas")
        if canvases:
            for c in canvases:
                try:
                    b = c.screenshot()
                    board_images.append(b)
                except Exception:
                    pass
    except Exception:
        pass

    if not board_images:
        # 找到可能的路单元素并截图
        divs = page.query_selector_all("div")
        for el in divs:
            try:
                box = el.bounding_box()
                if not box: continue
                if box["width"]>180 and box["height"]>80 and box["height"]<700:
                    try:
                        b = el.screenshot()
                        board_images.append(b)
                    except:
                        pass
                if len(board_images) >= 40:
                    break
            except Exception:
                continue

    if not board_images:
        full = page.screenshot(full_page=True)
        board_images.append(full)

    browser.close()
    return board_images

# ========== 主流程 ==========
def main():
    state = load_state()
    with sync_playwright() as p:
        try:
            board_imgs = try_load_dg_and_screenshot(p)
        except Exception as e:
            msg = f"❗ 无法抓取 DG 页面：{e}\n可能遇到滑块/反自动化或网络阻挡。需要人工介入。"
            print(msg)
            send_telegram_text(msg)
            return

    table_metrics=[]
    for img_bytes in board_imgs:
        m = analyze_board_image_bytes(img_bytes)
        if m is not None:
            table_metrics.append(m)

    result = classify_tables(table_metrics)
    label = result.get("label")
    now_ts = int(time.time())
    now_str = datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S")
    prev_status = state.get("status", "idle")
    prev_alert = state.get("alert_type")

    # 需要提醒的仅两种 label
    if label in ("放水", "中等胜率(中上)"):
        # 若之前不是 alert 或 alert 类型变化 -> 发送开始通知
        if prev_status != "running_alert" or prev_alert != label:
            state["status"] = "running_alert"
            state["alert_type"] = label
            state["start_ts"] = now_ts
            hist = state.get("history_minutes", [])
            if len(hist) >= 3:
                est = int(statistics.median(hist))
            elif len(hist) >= 1:
                est = int(statistics.mean(hist))
            else:
                est = 12
            est_end = now_ts + est*60
            est_end_str = datetime.fromtimestamp(est_end).strftime("%H:%M:%S")
            caption = (f"▶️ <b>检测到 放水/中高胜率（开始）</b>\n\n"
                       f"类型: {label}\n"
                       f"时间: {now_str}\n"
                       f"检测桌数: {result.get('total_tables')}，符合(满盘/长连)数: {result.get('cnt_full_like')}\n"
                       f"长龙(>=8): {result.get('cnt_long')}，超长龙(>=10): {result.get('cnt_superlong')}\n\n"
                       f"预计持续(基于历史估计): {est} 分钟\n预计结束(估计): {est_end_str}\n\n"
                       f"说明：使用你提供的判定规则（满盘长连 / 超长龙触发等）。")
            try:
                send_telegram_photo(board_imgs[0], caption)
            except Exception:
                send_telegram_text(caption)
            save_state(state)
            commit_state_git()
        else:
            print("仍处 alert 中，跳过重复通知。")
    else:
        # 非提醒时段
        if prev_status == "running_alert":
            start_ts = state.get("start_ts")
            duration_min = max(1, int((now_ts - (start_ts or now_ts))/60))
            hist = state.get("history_minutes", [])
            hist.append(duration_min)
            if len(hist) > 50: hist = hist[-50:]
            state["history_minutes"] = hist
            state["status"] = "idle"
            state["alert_type"] = None
            state["start_ts"] = None
            save_state(state)
            commit_state_git()
            caption = (f"⏸️ 放水/中高胜率 已结束\n结束时间: {now_str}\n实际持续: {duration_min} 分钟\n检测桌数: {result.get('total_tables')}\n已将本次持续加入历史用于以后估算。")
            send_telegram_text(caption)
        else:
            print("非提醒时段，未在 alert 中，未发送通知。")
            state["status"]="idle"; state["alert_type"]=None
            save_state(state)
            commit_state_git()

if __name__ == "__main__":
    main()

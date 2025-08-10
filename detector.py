# detector.py
# DG 自动检测 + Telegram 通知
# 要点：
#  - 使用 Playwright 自动打开 DG 页面并进入 Free 模式
#  - 截图整页，使用 OpenCV 对每个白色"桌面"区域做红/蓝密度分析与连区域启发式判断
#  - 根据你定义的规则判定 "放水" / "中等胜率(中上)" / "胜率中等" / "收割时段"
#  - 当发现 放水 或 中等胜率(中上) 时通过 Telegram 发消息
#  - 使用 state.json 在仓库内持久化放水开始时间并由 Actions commit 回仓库（需要 workflow 给 contents: write 权限）
#
# 注意：
#  - 脚本内的阈值/坐标检测基于启发式方法，不同分辨率或界面可能需要微调（见下方 PARAMETERS）
#  - 如果 DG 页面对自动化有强限制，可能需要调整 Playwright 的 userAgent/stealth 等（此处给出基础可用方案）
#
import os, sys, json, time, math, subprocess, datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import requests
import asyncio
from playwright.async_api import async_playwright

# -----------------------------
# 配置（如需改动，在此修改）
# -----------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "485427847")
DG_URLS = [ os.environ.get("DG_URL_1", "https://dg18.co/wap/"), os.environ.get("DG_URL_2", "https://dg18.co/") ]
CHECK_EVERY_MIN = 5   # GitHub Actions 由 schedule 控制，这里仅备份
STATE_PATH = Path("state.json")
# 图像检测参数（可能需要随着你页面分辨率微调）
MIN_TABLE_AREA = 3000        # 识别白框最小面积
RED_HSV_LOW = np.array([0, 80, 30])
RED_HSV_HIGH = np.array([12, 255, 255])
BLUE_HSV_LOW = np.array([90, 60, 30])
BLUE_HSV_HIGH = np.array([140, 255, 255])
# 连（长连/长龙）判断阈值（以单桌子垂直连的像素高度来近似）
LONG_CHAIN_HEIGHT_PX = 80  # 如果某色在竖方向上出现连续长块，视作长连/长龙的候选（需依据你界面微调）
DRAGON_COLS_THRESHOLD = 8  # 若单桌竖直连续数格（约）≥ 8，视作【长龙】
SUPER_DRAGON_COLS = 10     # 超长龙阈值（≥10）
# 整桌判断阈值
MIN_TABLES_FOR_PERCENT = 0.5   # ≥50% 桌面为放水（当用第1种规则）
MIN_DRAGON_TABLES = 3         # 至少3张桌子出现长龙/超长龙才为有效信号
# Telegram 图片文件名
TMP_SCREEN = "screen.png"

# -----------------------------
# 辅助：发送 Telegram 消息（支持图片）
# -----------------------------
def send_telegram_text(text: str, image_path: str = None):
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    base = f"https://api.telegram.org/bot{token}"
    # send text
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(base + "/sendMessage", data=data, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("Telegram text send failed:", e)
    # optionally send photo
    if image_path and os.path.exists(image_path):
        try:
            files = {"photo": open(image_path, "rb")}
            data = {"chat_id": chat_id, "caption": text}
            r = requests.post(base + "/sendPhoto", data=data, files=files, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print("Telegram photo send failed:", e)

# -----------------------------
# 状态保存与提交（用于跨 runs 追踪放水开始时间）
# 我们会在 workflow 中给 actions/checkout 权限并允许 contents: write，
# 这里脚本在检测到状态变化时会更新 state.json 并用 git 提交回仓库（使用内置 GITHUB_TOKEN）
# -----------------------------
def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except:
            pass
    return {"in_water": False, "start_ts": None, "last_seen": None}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    # commit back to repo so future runs can read
    # use GITHUB_TOKEN if available in env (Actions provides it)
    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("No GITHUB_TOKEN found; skipping git commit of state.json")
        return
    try:
        # configure git
        subprocess.run(["git", "config", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "add", str(STATE_PATH)], check=True)
        subprocess.run(["git", "commit", "-m", f"update state.json at {datetime.datetime.utcnow().isoformat()}"], check=False)
        # push using token
        origin_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + os.environ.get("GITHUB_REPOSITORY", "")
        if origin_url:
            repo_url = origin_url.replace("https://", f"https://x-access-token:{gh_token}@")
            subprocess.run(["git", "push", repo_url, "HEAD:refs/heads/HEAD"], check=False)
    except Exception as e:
        print("git commit/push failed (may still be okay):", e)

# -----------------------------
# 图像处理：识别页面上的“白色桌子框”并对每个区域统计红/蓝密度与垂直连长度
# 这是启发式方法，基于截图中“白色底 + 红/蓝圆圈” 可行
# -----------------------------
def analyze_screenshot(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("screenshot not found or unreadable")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 找到明显白色区域（桌面白框通常接近白色）
    lower_white = np.array([0,0,200])
    upper_white = np.array([255,40,255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # 腐蚀/膨胀以去小噪点
    kernel = np.ones((5,5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww*hh
        if area < MIN_TABLE_AREA:
            continue
        # 裁切桌面区域（加点 padding）
        pad = 4
        x1 = max(0, x-pad)
        y1 = max(0, y-pad)
        x2 = min(w, x+ww+pad)
        y2 = min(h, y+hh+pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tables.append({"rect": (x1,y1,x2,y2), "img": crop})

    # 如果没有找到明显白框，尝试以整个截图为一个大桌面（兼容不同UI）
    if not tables:
        tables.append({"rect": (0,0,w,h), "img": img})

    # 对每个桌面统计红/蓝像素密度 & 检测竖直长块（估计连长度）
    analysis = []
    for t in tables:
        crop = t["img"]
        ch, cw = crop.shape[:2]
        hsvc = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # red mask (two ranges possible; cover basic red)
        mask_r1 = cv2.inRange(hsvc:=hsvc if False else hsvc, np.array([0,60,30]), np.array([12,255,255]))
        mask_r2 = cv2.inRange(hsvc, np.array([170,60,30]), np.array([180,255,255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        mask_blue = cv2.inRange(hsvc, BLUE_HSV_LOW, BLUE_HSV_HIGH)
        red_count = int(np.count_nonzero(mask_red))
        blue_count = int(np.count_nonzero(mask_blue))
        total = ch*cw
        red_ratio = red_count/total
        blue_ratio = blue_count/total

        # 竖直投影，检测最大垂直连块（以连珠为竖向连续像素聚类近似）
        # 合并红蓝为 single mask for vertical runs detection but keep per-color
        def max_vertical_run(mask):
            # project mask vertically: for each column compute longest run of non-zero contiguous pixels
            max_run = 0
            cols = mask.shape[1]
            for col in range(cols):
                col_data = mask[:, col]
                # find longest consecutive non-zero
                curr = 0
                col_max = 0
                for v in col_data:
                    if v:
                        curr += 1
                        col_max = max(col_max, curr)
                    else:
                        curr = 0
                max_run = max(max_run, col_max)
            return max_run

        red_max_v = max_vertical_run(mask_red)
        blue_max_v = max_vertical_run(mask_blue)

        analysis.append({
            "rect": t["rect"],
            "red_ratio": red_ratio,
            "blue_ratio": blue_ratio,
            "red_count": red_count,
            "blue_count": blue_count,
            "red_max_v": red_max_v,
            "blue_max_v": blue_max_v,
            "w": cw,
            "h": ch
        })
    return analysis

# -----------------------------
# 根据每张桌面的分析结果应用你的规则判定：
#  - 计算：多少桌子有长连/长龙（依据 red_max_v 或 blue_max_v）
#  - 计算：桌面“饱满度”（非白色区域/颜色密度）
#  - 判定放水、中等胜率、中等、收割
# -----------------------------
def classify_overall(analysis):
    total_tables = len(analysis)
    dragon_tables = 0
    super_dragon_tables = 0
    long_chain_tables = 0
    full_score_tables = 0  # 桌面密度高（红或蓝密度高）

    for a in analysis:
        # 判定是否为 “长龙/超长龙” 根据 red_max_v / blue_max_v，相对桌高比例
        h = a["h"]
        # 把像素阈值转换为“格数”估计：这里用实际 px thresholds
        if a["red_max_v"] >= SUPER_DRAGON_COLS or a["blue_max_v"] >= SUPER_DRAGON_COLS:
            super_dragon_tables += 1
            dragon_tables += 1
        elif a["red_max_v"] >= DRAGON_COLS_THRESHOLD or a["blue_max_v"] >= DRAGON_COLS_THRESHOLD:
            dragon_tables += 1
        # 判定长连（较短的长连）
        if (a["red_max_v"] >= (DRAGON_COLS_THRESHOLD//2)) or (a["blue_max_v"] >= (DRAGON_COLS_THRESHOLD//2)):
            long_chain_tables += 1
        # 饱满度（简单以颜色比率）
        if (a["red_ratio"] + a["blue_ratio"]) > 0.006:   # 经验阈值，需要根据截屏分辨率微调
            full_score_tables += 1

    # 规则一：满桌长连/长龙类型（≥50% 桌子为“饱满/长连/长龙”）
    percent_full = full_score_tables / max(1, total_tables)
    rule1 = percent_full >= MIN_TABLES_FOR_PERCENT

    # 规则二：超长龙 + 多张长龙
    rule2 = (super_dragon_tables >= 1) and ((dragon_tables - super_dragon_tables) >= 2)

    # classify according to your priority:
    # - if rule1 or rule2 and dragon_tables >= MIN_DRAGON_TABLES => 放水
    # - else if mixed (有2桌长龙以上但不满足>=50%饱满) => 中等胜率（中上）
    # - else if many空桌、单跳占多数 => 收割（胜率调低）或 胜率中等
    is_water = False
    is_mid_high = False
    is_mid = False
    is_low = False

    if (rule1 or rule2) and (dragon_tables >= MIN_DRAGON_TABLES):
        is_water = True
    else:
        # 如果有 >=2 桌长龙，且占比不够 50%，判为中等胜率（中上）
        if dragon_tables >= 2:
            is_mid_high = True
        # 若没有多数饱满且 dragon_tables 很少，判为胜率中等/收割
        if dragon_tables < 2 and percent_full < 0.2:
            # 判断为收割（胜率调低）
            is_low = True
        else:
            is_mid = True

    result = {
        "total_tables": total_tables,
        "dragon_tables": dragon_tables,
        "super_dragon_tables": super_dragon_tables,
        "full_score_tables": full_score_tables,
        "percent_full": percent_full,
        "rule1": rule1,
        "rule2": rule2,
        "is_water": is_water,
        "is_mid_high": is_mid_high,
        "is_mid": is_mid,
        "is_low": is_low,
    }
    return result

# -----------------------------
# 高阶判断：依据你的入场策略（状况A）做更细粒度判断（启发式）
# 这里我们示范：如果单桌出现“断连开单”型（例如长连后断后有单）则视作符合状况A（需提醒）
# 真实的“断连开单”模式在图像上要用更多历史帧来判断；这里用当前截图内连续列检测近似
# -----------------------------
def detect_profitable_tables(analysis):
    # 返回符合状况A的桌子索引列表（示范）
    profitable = []
    for idx, a in enumerate(analysis):
        # 判定：若某一颜色在竖直方向上出现超长串（>= SUPER_DRAGON_COLS）且该桌还有次级的断连结构（简单用另一颜色的小连判断）
        if a["red_max_v"] >= SUPER_DRAGON_COLS or a["blue_max_v"] >= SUPER_DRAGON_COLS:
            # 进一步检查另一颜色是否存在短跳（作为断连开单的判定）
            if (a["red_max_v"] >= SUPER_DRAGON_COLS and a["blue_max_v"] <= 3) or (a["blue_max_v"] >= SUPER_DRAGON_COLS and a["red_max_v"] <= 3):
                profitable.append(idx)
            else:
                # 也可接受长连然后短断再开回的形式
                profitable.append(idx)
        # 另：若存在明显“多连”结构（中等竖连）也视作可入场参考
        elif a["red_max_v"] >= DRAGON_COLS_THRESHOLD or a["blue_max_v"] >= DRAGON_COLS_THRESHOLD:
            profitable.append(idx)
    return profitable

# -----------------------------
# Playwright 自动化：打开 DG 页面，点击 Free，并截图
# -----------------------------
async def capture_dg_screenshot(save_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"], headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        )
        page = await context.new_page()
        # 尝试两个 URL 中的一个能进的
        success = False
        for url in DG_URLS:
            try:
                await page.goto(url, timeout=45000)
                success = True
                break
            except Exception as e:
                print("goto failed:", e)
        if not success:
            raise RuntimeError("Cannot open DG URLs")

        # 等待页面稳定
        await page.wait_for_timeout(3000)

        # 尝试点击 'Free' / '免费试玩' 按钮（多个语言/样式）
        # 我们尝试几种常见文本或按钮样式
        selectors = [
            "text=Free", "text=免费试玩", "text=免费", "button:has-text('Free')", "button:has-text('免费')"
        ]
        clicked = False
        for s in selectors:
            try:
                el = await page.query_selector(s)
                if el:
                    await el.click(timeout=3000)
                    clicked = True
                    break
            except Exception:
                pass
        # 有些站点需要滑动安全条（滑动条可能是一个 input range）
        # 尝试滚动页面以触发加载
        await page.mouse.wheel(0, 1000)
        await page.wait_for_timeout(2000)
        await page.mouse.wheel(0, -200)
        await page.wait_for_timeout(2000)

        # 等待若干秒让桌面加载
        await page.wait_for_timeout(5000)

        # 另外尝试点击 hall 或进入 game area if exists
        # 尝试截图整页
        await page.screenshot(path=save_path, full_page=True)
        await context.close()
        await browser.close()
        return True

# -----------------------------
# 主流程
# -----------------------------
def ts_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()  # Malaysia +8

async def main_async():
    print("Starting DG detection at", ts_now())
    # 1) capture screenshot
    try:
        await capture_dg_screenshot(TMP_SCREEN)
    except Exception as e:
        print("Capture failed:", e)
        send_telegram_text(f"DG monitor: 无法打开 DG 页面或截图失败：{e}")
        return

    # 2) analyze screenshot
    try:
        analysis = analyze_screenshot(TMP_SCREEN)
        overall = classify_overall(analysis)
        profitable = detect_profitable_tables(analysis)
    except Exception as e:
        print("Analyze failed:", e)
        send_telegram_text(f"DG monitor: 截图分析失败：{e}")
        return

    # 3) load state
    state = load_state()

    # 4) decide actions based on classification
    now = datetime.datetime.now(datetime.timezone.utc)
    summary = f"检测时间 (MYT): {datetime.datetime.now().astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"总桌数: {overall['total_tables']}, 长龙桌: {overall['dragon_tables']}, 超长龙: {overall['super_dragon_tables']}, 饱满桌: {overall['full_score_tables']}\n"
    summary += f"percent_full: {overall['percent_full']:.2f}, rule1: {overall['rule1']}, rule2: {overall['rule2']}\n"

    if overall["is_water"]:
        # 放水时段：必须提醒（若之前已经在放水中则只更新 last_seen，不重复发多次提醒）
        if not state.get("in_water", False):
            # 新开始的放水
            state["in_water"] = True
            state["start_ts"] = now.isoformat()
            state["last_seen"] = now.isoformat()
            save_state(state)
            # build message
            msg = f"🚨 放水时段（提高胜率）检测到！\n{summary}\n符合放水规则（rule1/或 rule2）。\n符合可盈利桌: {len(profitable)} 张（索引）。请马上手动入场。"
            # attach screenshot
            send_telegram_text(msg, TMP_SCREEN)
        else:
            # 已在放水中，更新 last_seen & 不重复提醒
            state["last_seen"] = now.isoformat()
            save_state(state)
            # 不必每次都发提醒；可发简短更新（这里选择不发送以避免炸群）
            print("Still in water; updated last_seen.")
    elif overall["is_mid_high"]:
        # 中等胜率（中上） -> 小提醒（只在首次进入时通知）
        if state.get("in_mid_high") != True:
            state["in_mid_high"] = True
            state["mid_high_start"] = now.isoformat()
            save_state(state)
            msg = f"🔔 中等胜率（中上）检测到 — 小提醒。\n{summary}\n说明：局面接近放水但不完全。"
            send_telegram_text(msg, TMP_SCREEN)
        else:
            # 已处在中等胜率中，只更新时间
            state["mid_high_last"] = now.isoformat()
            save_state(state)
            print("Still mid-high; updated state.")
    else:
        # 中等或收割时段（不提醒） -> 若之前处在放水/中上则发放水结束通知
        if state.get("in_water", False):
            # 放水刚结束，计算持续时间
            start = datetime.datetime.fromisoformat(state.get("start_ts"))
            end = now
            dur = end - start
            mins = int(dur.total_seconds() / 60)
            msg = f"✅ 放水已结束。\n开始时间(MYT): {start.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n结束时间(MYT): {end.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}\n持续: {mins} 分钟\n{summary}"
            send_telegram_text(msg, TMP_SCREEN)
            # 清除状态
            state["in_water"] = False
            state["start_ts"] = None
            state["last_seen"] = None
            save_state(state)
        # 清除中上状态
        if state.get("in_mid_high", False):
            state["in_mid_high"] = False
            state["mid_high_start"] = None
            state["mid_high_last"] = None
            save_state(state)

    # optional logging
    print("Overall:", overall)
    print("Profitable count:", len(profitable))

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
DG 自动监测脚本（用于 GitHub Actions）
功能：
- 打开 DG 网站（https://dg18.co/ 或 https://dg18.co/wap/），点击 Free/免费试玩，并模拟滑动安全条
- 抓取每个桌面（以 DOM 优先，找不到则截图）
- 对每张桌面进行图像/格子分析，识别长连/多连/长龙/超长龙/单跳等
- 根据您设定的规则做全局判定（放水、中等胜率、胜率中等、收割）
- 当进入放水/中等胜率时通过 Telegram 发送通知；放水结束时发送结束通知并报告持续时间
- 保存状态到 status.json 并 commit 回仓库（用于跨次 workflow 的状态保持）
"""

import os
import json
import time
import math
import requests
from datetime import datetime, timezone, timedelta

# If running locally or without secrets, fallback to defaults below:
DEFAULT_TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
DEFAULT_TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "485427847")

TELEGRAM_BOT_TOKEN = DEFAULT_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = DEFAULT_TELEGRAM_CHAT_ID

# DG links (as you required)
DG_LINKS = [
    "https://dg18.co/",
    "https://dg18.co/wap/"
]

# 判定阈值（严格按照你定义）
THRESHOLDS = {
    # 放水（满盘长连局势）判断：若总桌≥20 则 >=8 符合；若总桌≥10 则 >=4 符合
    "full_table_count_20_need": 8,
    "full_table_count_10_need": 4,
    # 中等胜率（中上）判断：检测到 20 张桌子时至少 6 张符合；10 张时至少 3 张符合
    "mid_high_need_20": 6,
    "mid_high_need_10": 3,
    # 超长龙触发型：1 超长龙 + 至少 2 条长龙（总共 >=3 张桌）
    "super_dragon_need": 1,
    "dragon_need": 2,
    # 全局判断时“至少 2 桌有长龙或超长龙”也会作为条件
    "mid_high_min_dragons": 2,
    # 分类“连”定义（基于你给出）
    "long_chain_len": 4,
    "dragon_len": 8,
    "super_dragon_len": 10
}

# 状态文件路径（被 workflow commit 回仓库）
STATUS_FILE = "status.json"

# 时区：马来西亚 (UTC+8)
LOCAL_TZ = timezone(timedelta(hours=8))

# ---------- 辅助：发送 Telegram ----------
def send_telegram(text, bot_token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    resp = requests.post(url, data=data, timeout=20)
    return resp.status_code, resp.text

# ---------- 状态文件读写 ----------
def read_status():
    if not os.path.exists(STATUS_FILE):
        return {"state": "idle", "start_time": None}
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_status(st):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

# ---------- DG 访问与抓取（使用 Playwright） ----------
def capture_boards_and_analyze():
    """
    使用 Playwright 自动打开 DG、点击 Free、滑动安全条、等待牌面加载，
    并尝试抓取每张桌子的 DOM 或截图进行分析。
    返回结构：列表 of { 'table_id': str, 'analysis': { 'max_run': int, 'type': 'LONG/DRAGON/...', ... } }
    """
    from playwright.sync_api import sync_playwright
    import numpy as np
    from PIL import Image
    import cv2
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        )
        page = context.new_page()
        # 逐一尝试 DG 链接，直到可进站
        open_success = False
        for link in DG_LINKS:
            try:
                page.goto(link, timeout=30000)
                open_success = True
                break
            except Exception as e:
                print("Open failed:", link, e)
        if not open_success:
            raise RuntimeError("无法打开 DG 任何入口。")

        # 页面里可能有“Free/免费试玩”按钮，这里尝试点击并滑动安全条。
        # 注意：不同版本页面结构不同，以下尝试多种选择器。
        time.sleep(2)
        try:
            # 尝试找到 Free/免费试玩按钮并点击
            for sel in ["text=Free", "text=免费试玩", "button:has-text('Free')", "button:has-text('免费')"]:
                try:
                    if page.locator(sel).count() > 0:
                        page.locator(sel).first.click(timeout=3000)
                        time.sleep(1)
                        break
                except:
                    continue
            # 等待跳转/弹出新页面
            time.sleep(2)
        except Exception as e:
            print("点击 Free 可能失败：", e)

        # 模拟滑动安全条（常见做法：找到滑块并拖动）
        try:
            # 尝试找到滑动元素
            # 这里用 JS 尝试查找常见滑动条 class/id
            page.wait_for_timeout(1000)
            slider_found = False
            for attempt_sel in ["#nc_1_n1z", ".slider", ".drag", ".verify-slider", "div[role='slider']"]:
                try:
                    if page.locator(attempt_sel).count() > 0:
                        # 拖动
                        box = page.locator(attempt_sel).bounding_box()
                        if box:
                            x = box["x"] + box["width"] / 2
                            y = box["y"] + box["height"] / 2
                            # drag by mouse
                            page.mouse.move(x, y)
                            page.mouse.down()
                            page.mouse.move(x + 300, y, steps=15)
                            page.mouse.up()
                            slider_found = True
                            time.sleep(1.2)
                            break
                except Exception:
                    continue
            if not slider_found:
                # 有时页面会自动跳转或者没有滑块
                pass
        except Exception as e:
            print("滑动安全条步骤遇到问题：", e)

        # 等待牌面加载（这段时间应足够让所有 table render）
        page.wait_for_timeout(4000)

        # 尝试从 DOM 抓取桌面列表：找寻每个 game 框的容器
        # 常见可能的选择器（根据实际可能需要调整）
        board_selectors = [
            ".game-list .game-item", ".table-list .table", ".gameBox", ".bet-table", ".game-item"
        ]

        tables = []
        for sel in board_selectors:
            try:
                items = page.locator(sel)
                if items.count() > 0:
                    # 取每一个 item 的截图（元素截图）或 DOM innerHTML
                    for i in range(items.count()):
                        el = items.nth(i)
                        # 尝试拿到一个 identifier：table name 或 id
                        table_id = None
                        try:
                            table_id = el.get_attribute("id")
                        except:
                            table_id = f"{sel}-{i}"
                        # 先尝试解析 DOM 中的格子信息（例如：每个格子可能是 <div class='dot b'> 或 img)
                        inner = ""
                        try:
                            inner = el.inner_html()
                        except:
                            inner = ""
                        # 截图备用（元素截图）
                        try:
                            path = f"/tmp/table_{i}.png"
                            el.screenshot(path=path)
                        except Exception:
                            # 回退为整体页面截图并裁切（这里简单保存整页）
                            path = f"/tmp/page_snap.png"
                            page.screenshot(path=path, full_page=True)
                        tables.append({"id": table_id or f"table-{i}", "html": inner, "screenshot": path})
                    break
            except Exception:
                continue

        # 如果上述没抓到任何 tables，则尝试按照常见页面的“房间卡片”选择器抓取
        if not tables:
            # 尝试更直接地抓取所有<img>或canvas并存为单张图片供分析
            page.screenshot(path="/tmp/full_page.png", full_page=True)
            # fallback - treat full page as single board
            tables.append({"id": "fullpage", "html": "", "screenshot": "/tmp/full_page.png"})

        # 对每张截图进行图像分析（检测红圈/蓝圈/连续run）
        for t in tables:
            analysis = analyze_board_image(t["screenshot"])
            results.append({"table_id": t["id"], "analysis": analysis})

        browser.close()
    return results

# ---------- 图像分析：从桌子截图识别格子与颜色分布 ----------
def analyze_board_image(img_path):
    """
    对截图进行处理，尽量找出格子里红/蓝圆点的分布（简化：找到大颗红/蓝点并估计连的长度）
    返回示例：
    {
        "max_run_same_side": 9,
        "runs": [ {"side":"B","len":9}, ... ],
        "has_long_chain": True / False,
        "has_dragon": True / False,
        "has_super_dragon": False / True,
        "dominant": "B"/"P"/None,
        "single_jumps_count": n,
        ...
    }
    (注：B 表示庄 (red), P 表示闲 (blue))
    """
    import cv2
    import numpy as np
    from PIL import Image

    # 读取图像
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path)
    except Exception:
        img = cv2.imread(img_path)

    if img is None:
        return {"error": "cannot_read_image"}

    h, w = img.shape[:2]

    # 将图片缩小以加速处理（保留比例）
    scale = 1.0
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色（庄）与蓝色（闲）阈值（可微调）
    lower_red1 = np.array([0, 60, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 40])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 腐蚀/膨胀去噪
    kernel = np.ones((3,3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    # 找轮廓作为“点”的检测
    contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_centers = []
    blue_centers = []
    for c in contours_r:
        area = cv2.contourArea(c)
        if area < 30:  # noise threshold
            continue
        (x,y),r = cv2.minEnclosingCircle(c)
        red_centers.append((int(x), int(y), int(r)))
    for c in contours_b:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        (x,y),r = cv2.minEnclosingCircle(c)
        blue_centers.append((int(x), int(y), int(r)))

    # 合并点并按 x（或 y）排序以估算“连”情况
    # 这里简化把“同排连续”视为 x 差距小（或按列/row检测需具体页面布局）
    points = []
    for x,y,r in red_centers:
        points.append({"side":"B","x":x,"y":y})
    for x,y,r in blue_centers:
        points.append({"side":"P","x":x,"y":y})

    if not points:
        return {
            "max_run_same_side": 0,
            "runs": [],
            "has_long_chain": False,
            "has_dragon": False,
            "has_super_dragon": False,
            "dominant": None,
            "single_jumps_count": 0
        }

    # 简略估算：按 x 坐标排序并统计连续相同 side 的最长 run
    points_sorted = sorted(points, key=lambda p: (p["x"], p["y"]))
    runs = []
    cur_side = points_sorted[0]["side"]
    cur_len = 1
    for p in points_sorted[1:]:
        if p["side"] == cur_side:
            cur_len += 1
        else:
            runs.append({"side": cur_side, "len": cur_len})
            cur_side = p["side"]
            cur_len = 1
    runs.append({"side": cur_side, "len": cur_len})

    max_run = max(r["len"] for r in runs) if runs else 0

    has_long_chain = max_run >= THRESHOLDS["long_chain_len"]
    has_dragon = max_run >= THRESHOLDS["dragon_len"]
    has_super_dragon = max_run >= THRESHOLDS["super_dragon_len"]

    # 单跳：以 runs 中大多数为 1 的次数计
    single_jumps_count = sum(1 for r in runs if r["len"] == 1)
    # dominant side
    count_B = sum(1 for r in runs if r["side"]=="B")
    count_P = sum(1 for r in runs if r["side"]=="P")
    dominant = "B" if count_B > count_P else ("P" if count_P > count_B else None)

    return {
        "max_run_same_side": max_run,
        "runs": runs,
        "has_long_chain": has_long_chain,
        "has_dragon": has_dragon,
        "has_super_dragon": has_super_dragon,
        "dominant": dominant,
        "single_jumps_count": single_jumps_count,
        "red_count": len(red_centers),
        "blue_count": len(blue_centers)
    }

# ---------- 全局判定函数 ----------
def evaluate_global_state(tables_analysis):
    """
    基于你定义的规则，对所有桌面分析结果汇总并返回一个全局状态：
    - "放水" (strong)
    - "中等勝率_中上" (mid_high)
    - "勝率中等" (no_alert)
    - "收割" (no_alert)
    同时返回用于 Telegram 的 summary 文本与估算持续时间（粗略）
    """
    total_tables = len(tables_analysis)
    count_long_like = 0  # 符合“满盘长连”条件的单桌计数（长连或长龙）
    count_dragon = 0
    count_super_dragon = 0
    count_multilen = 0  # 有多连/连珠等
    single_jump_tables = 0

    for t in tables_analysis:
        a = t.get("analysis", {})
        if a.get("has_long_chain"):
            count_long_like += 1
        if a.get("has_dragon"):
            count_dragon += 1
        if a.get("has_super_dragon"):
            count_super_dragon += 1
        # 多连/连珠：这里以 runs 中存在 len>=4 且存在多段为判断（简化）
        if sum(1 for r in a.get("runs", []) if r["len"] >= THRESHOLDS["long_chain_len"]) >= 2:
            count_multilen += 1
        if a.get("single_jumps_count", 0) >= 3:
            single_jump_tables += 1

    # 满盘长连局势型判定
    is_full_long = False
    if total_tables >= 20 and count_long_like >= THRESHOLDS["full_table_count_20_need"]:
        is_full_long = True
    if total_tables >= 10 and total_tables < 20 and count_long_like >= THRESHOLDS["full_table_count_10_need"]:
        is_full_long = True

    # 超长龙触发型
    is_super_trigger = False
    if count_super_dragon >= THRESHOLDS["super_dragon_need"] and count_dragon >= THRESHOLDS["dragon_need"]:
        if (count_super_dragon + count_dragon) >= 3:
            is_super_trigger = True

    # 中等胜率（中上）判定条件（融合）
    is_mid_high = False
    if total_tables >= 20 and (count_long_like + count_dragon + count_multilen) >= THRESHOLDS["mid_high_need_20"]:
        is_mid_high = True
    if total_tables >= 10 and total_tables < 20 and (count_long_like + count_dragon + count_multilen) >= THRESHOLDS["mid_high_need_10"]:
        is_mid_high = True
    if (count_dragon + count_super_dragon) >= THRESHOLDS["mid_high_min_dragons"]:
        is_mid_high = True

    # 现在按优先级判定最终全局状态
    if is_full_long or is_super_trigger:
        state = "放水"
    elif is_mid_high:
        state = "中等勝率_中上"
    else:
        # 判断是胜率中等或收割（以空桌 & 单跳多来区分）
        # 简化逻辑：如果大多数桌子单跳/空白 -> 收割
        if single_jump_tables >= (total_tables * 0.6):
            state = "收割"
        else:
            state = "勝率中等"

    # 构建 summary 文本用于 Telegram
    summary = (
        f"DG 全局检测结果：{state}\n"
        f"总桌数：{total_tables}\n"
        f"长连/多连类桌数：{count_long_like}\n"
        f"长龙数量：{count_dragon}\n"
        f"超长龙数量：{count_super_dragon}\n"
        f"多连/连珠桌数(估算)：{count_multilen}\n"
        f"单跳桌数(>=3 单跳计)：{single_jump_tables}\n"
    )

    # 估算剩余持续时间（粗略）：如果是放水或中等胜率，依据是否有超长龙/长龙多
    est_minutes = None
    if state == "放水":
        # 更多长龙 -> 预计持续更长（经验值）
        base = 8
        extra = min(30, (count_dragon + count_super_dragon) * 3)
        est_minutes = base + extra
    elif state == "中等勝率_中上":
        base = 5
        extra = min(20, (count_dragon + count_multilen) * 2)
        est_minutes = base + extra

    return {
        "state": state,
        "summary": summary,
        "est_minutes": est_minutes,
        "details": {
            "total": total_tables,
            "count_long_like": count_long_like,
            "count_dragon": count_dragon,
            "count_super_dragon": count_super_dragon,
            "count_multilen": count_multilen,
            "single_jump_tables": single_jump_tables
        }
    }

# ---------- 主运行逻辑 ----------
def main():
    # 读取上一次状态
    prev_status = read_status()
    prev_state = prev_status.get("state", "idle")
    prev_start = prev_status.get("start_time", None)

    # 捕获桌子并分析
    try:
        tables = capture_boards_and_analyze()
    except Exception as e:
        print("抓取或分析 DG 时出错：", e)
        # 在失败时不改变状态；可发送一条错误日志到 Telegram（可选）
        send_telegram(f"DG 监测脚本错误：{e}")
        return

    # 统计并判定全局
    eval_res = evaluate_global_state(tables)
    state = eval_res["state"]
    summary = eval_res["summary"]
    est_min = eval_res["est_minutes"]

    now = datetime.now(LOCAL_TZ)

    # 若当前为'放水'或'中等勝率_中上'，而之前不是，则发开始通知并记录 start_time
    if state in ("放水", "中等勝率_中上") and prev_state not in ("放水", "中等勝率_中上"):
        start_time = now.isoformat()
        new_status = {"state": state, "start_time": start_time}
        write_status(new_status)
        # 发送 Telegram 开始通知（含预计结束时间）
        if est_min:
            est_end = now + timedelta(minutes=est_min)
            remain_text = f"预计放水结束时间（本地UTC+8）：{est_end.strftime('%Y-%m-%d %H:%M:%S')}，估计剩余约 {est_min} 分钟。"
        else:
            remain_text = "预计持续时间不可估算。"
        text = f"🔔 *进入放水/中等胜率提醒*\n状态：*{state}*\n\n{summary}\n{remain_text}\n\n(此通知由自动监测系统发出)"
        send_telegram(text)
        print("已发送开始提醒。")

    # 若之前为放水/中上，但现在变成非放水（结束），则发送结束通知并计算持续时间
    elif prev_state in ("放水", "中等勝率_中上") and state not in ("放水", "中等勝率_中上"):
        # 计算持续时间
        start_time = prev_status.get("start_time")
        if start_time:
            st = datetime.fromisoformat(start_time)
            duration = now - st
            minutes = int(duration.total_seconds() / 60)
        else:
            minutes = None
        # 更新状态为 idle / state
        new_status = {"state": state, "start_time": None}
        write_status(new_status)
        # 发送结束通知
        if minutes is not None:
            text = f"⛔️ 放水已结束\n先前状态：{prev_state}\n本次放水持续时间：{minutes} 分钟\n当前全局状态：{state}\n\n{summary}"
        else:
            text = f"⛔️ 放水已结束（无开始时间记录）\n当前全局状态：{state}\n\n{summary}"
        send_telegram(text)
        print("已发送结束提醒。")

    else:
        # 状态未发生变化：如果仍在放水/中上，可选择不重复发送（按你要求）
        print("状态无变化：", state)
        # 我们在需要时也可以发送周期性“仍在放水”通知（这里默认不发送以免刷屏）
        # 同时保存当前状态（保持 start_time）
        if prev_state in ("放水", "中等勝率_中上"):
            # 保持原记录
            write_status(prev_status)
        else:
            write_status({"state": state, "start_time": None})

    # 提示：将 status.json commit 回仓库由 workflow 后段处理（workflow yaml 提供此步骤）
    print("本次检测完成。")

if __name__ == "__main__":
    main()

DG Monitor - Setup (简短步骤)

1) 在 GitHub 上建立一个 public repository（或私有但需付费分钟数注意）。
2) 在仓库中创建以上文件:
   - requirements.txt
   - main.py
   - .github/workflows/monitor.yml

3) 在仓库 Settings -> Secrets -> Actions 中新增 Secrets:
   - TG_TOKEN    -> 你 Telegram Bot Token (例如: 8134230045:AAH6C_...)
   - TG_CHAT_ID  -> 你的 Chat ID (例如: 485427847)
   - DG_URLS     -> DreamGaming 链接, 例如: https://dg18.co,https://dg18.co/wap/

4) 确保仓库的 Actions 权限: Settings -> Actions -> General -> Workflow permissions -> 选 "Read and write permissions".
5) 提交 (commit & push) 所有文件到仓库。
6) 在 Actions 页面运行 "DG Monitor" 工作流一次（Use "Run workflow"）以便立刻检测并生成 state.json。
7) 观察 Actions 日志来调试（首次运行可能需调整阈值）。每 5 分钟自动运行一次。

注意:
- 若发现 Telegram 没收到通知，请检查 Actions logs, 并检查是否正确设置 Secrets。
- 若脚本因为页面验证码或 anti-bot 被阻挡，日志会显示错误并保存截图供调试。

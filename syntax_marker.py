# syntax_marker.py
def label_action(
    prev_action, current_action, 
    reward, prev_reward, 
    action_window, reward_window,
    reward_threshold=2.0,
    slope_threshold=0.5,
    stability_low=0.3,
    stability_high=0.8
):
    """
    根據策略穩定度與reward變化判定智慧狀態標籤（∅ Σ Δ Ω）

    Args:
        prev_action (int): 上一步行為
        current_action (int): 當前行為
        reward (float): 當前回饋值
        prev_reward (float): 前一步回饋值
        action_window (List[int]): 最近幾步的行動記錄
        reward_window (List[float]): 最近幾步的reward記錄
        reward_threshold (float): Ω的總reward判定門檻
        slope_threshold (float): Δ判定的reward變化率
        stability_low (float): ∅門檻的低穩定界線
        stability_high (float): Σ/Ω判定的高穩定界線

    Returns:
        str: 標籤（∅, Σ, Δ, Ω）
    """

    # 預設防呆
    if not action_window:
        action_window = [current_action]
    if not reward_window:
        reward_window = [reward]

    # 策略穩定度：過去幾步與現在一致的比例
    same_action_count = sum([a == current_action for a in action_window])
    policy_stability = same_action_count / max(1, len(action_window))

    # reward變化率（斜率）
    reward_slope = reward - prev_reward

    # reward總累積
    cumulative_reward = sum(reward_window)

    # 標記邏輯判定
    if reward == 0 and policy_stability < stability_low:
        return '∅'  # 潛能探索期
    elif policy_stability > stability_high and abs(reward_slope) < 0.01:
        return 'Σ'  # 穩定存在
    elif abs(reward_slope) > slope_threshold:
        return 'Δ'  # 震盪分裂
    elif cumulative_reward > reward_threshold and policy_stability > stability_high:
        return 'Ω'  # 智慧整合
    else:
        return 'Δ'  # 默認為分裂

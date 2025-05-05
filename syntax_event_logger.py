# syntax_event_logger.py（簡化版本）

import json

class SyntaxEventLogger:
    def __init__(self, h_threshold=0.8, s_threshold=0.5, q_threshold=0.3):
        self.h_threshold = h_threshold
        self.s_threshold = s_threshold
        self.q_threshold = q_threshold
        self.event_log = []
        self.prev_h = None
        self.prev_s = None
        self.prev_event = None

    def log_event(self, episode, h, s, q, terminated):
        event = None

        if h < 0.5 and s < 0.2:
            event = "∅"
        elif 0.5 <= h < self.h_threshold:
            event = "Σ"
        elif h >= self.h_threshold:
            event = "Δ"
        if terminated and s >= self.s_threshold:
            event = "Ω"

        if self.prev_s is not None and s < self.prev_s:
            event = "⊖"
        if self.prev_h is not None and h > self.prev_h and self.prev_event == "⊖":
            event = "≈"

        if event:
            self.event_log.append({
                "episode": int(episode),
                "event": event,
                "h": float(h),
                "s": float(s),
                "q": float(q)
            })

        self.prev_h = h
        self.prev_s = s
        self.prev_event = event

    def save_log(self, save_path="wisdom_rhythm.json"):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.event_log, f, indent=4, ensure_ascii=False)
        print(f"✅ 智慧節奏日誌已儲存到 {save_path}")

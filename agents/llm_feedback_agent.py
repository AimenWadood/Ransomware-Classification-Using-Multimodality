# agents/llm_feedback_agent.py
from models.local_phi import LocalPhiModel

class LLMFeedbackAgent:
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.llm = LocalPhiModel(model_id)

    def give_feedback(self, y_true, y_pred, epoch: int):
        # small summary to keep prompts short
        acc = (y_true == y_pred).mean()
        prompt = (
            f"Epoch {epoch+1} results.\n"
            f"Accuracy: {acc:.4f}.\n"
            "Suggest 2-3 concrete training tweaks (class balancing, LR, epochs, model choice)."
        )
        try:
            reply = self.llm.generate_reply(prompt, max_new_tokens=200)
            print(f"\n LLM feedback:\n{reply}\n")
            return reply
        except Exception as e:
            # never block training if LLM fails
            print(f"[LLM feedback disabled due to error: {e}]")
            return "Fallback: consider tuning class balance, learning rate, and training epochs."

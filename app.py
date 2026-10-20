"""
Gradio demo for the SMS spam filter.

Run locally with `python app.py`, then open the URL it prints.
This is also the entry point for Hugging Face Spaces.
"""
import gradio as gr

from spam_filter import load_model, predict_label

text_clf = load_model()

EXAMPLES = [
    "Hey, are we still on for lunch tomorrow?",
    "Your package is out for delivery and should arrive by 6pm.",
    "You won 2 free tickets to the Super Bowl text us to claim your prize.",
    "URGENT! Your mobile number has won a £2000 prize. Call 09061701461 to claim now.",
    "Thanks for registering. Text 4343 to receive free updates on medicare.",
]


def classify(text):
    text = (text or "").strip()
    if not text:
        return "Type or paste a message to check it."
    if predict_label(text_clf, text) == "spam":
        return "Spam: this message looks like spam and would be filtered out."
    return "Real message: this message looks legitimate and would reach your inbox."


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Text message", placeholder="Paste a text message here"),
    outputs=gr.Textbox(label="Result"),
    examples=EXAMPLES,
    title="SMS Spam Filter",
    description=(
        "An AI filter that sorts incoming texts into real and spam so your team only "
        "sees messages that matter. Try one of the examples or paste your own message."
    ),
    article=(
        "Trained on the public UCI SMS Spam Collection (5,572 English text messages). "
        "It is a demo: a person should still review anything important."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(show_error=True)

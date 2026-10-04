"""Start a GPT Realtime session with text conversation history."""

from bench import interact
from bench.protocol import event_context
from data_processing.context import text_message_event

MODEL = "gpt-realtime-fast"
CONVERSATION = [
    ("user", "I am planning a desk and I prefer birch wood. can you give a recommendation"),
    # ("assistant", "Birch is a good choice for a light, clean-looking desk."),
]


def context():
    events = [text_message_event(role, text) for role, text in CONVERSATION]
    return event_context(events)


if __name__ == "__main__":
    print("Completed turns will be preloaded without requesting a response.")
    print("Speak a new turn and pause mid-sentence to test whether the model waits.\n")
    interact.run(model=MODEL, context=context())

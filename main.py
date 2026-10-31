"""Command line demo of the HR assistant
Run with: main.py
"""

from hr_policy_rag_assistant.pipeline import ask_agent, build_hr_assistant


def main():
    print("Building the HR Policy Assistant..")
    agent = build_hr_assistant()
    print("Assistant Ready.\n")

    demo_questions = [
        "How many paid annual leaves do i get?",
        "What is the notice period during probation?",
        "Can i work from home every day?",
    ]

    for question in demo_questions:
        print("=" * 60)
        print(f"QUESTION: {question}")
        print("-" * 60)
        answer = ask_agent(agent, question)
        print(f"ANSWER: {answer}")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()

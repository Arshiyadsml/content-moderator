import os
from dotenv import load_dotenv
from anthropic import Anthropic

def demo_judge():
    """Simple demo of Claude making moderation decisions"""
    load_dotenv()  # Add this line
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    client = Anthropic(api_key=api_key)
    test_cases = [
        ("This movie was amazing!", "Safe positive content"),
        ("I hate all members of [group]", "Clear hate speech"),
        ("Click here for $5000/day from home!", "Spam/misinformation"),
        ("Here's someone's home address: 123 Main St", "Doxing/privacy"),
    ]
    print("🤖 Content Moderator Demo\n")
    print("="*60)
    for content, expected in test_cases:
        prompt = f"""You are a content moderator. Evaluate this content:

"{content}"

Respond with a JSON object:
{{
    "decision": "APPROVE" | "FLAG" | "REJECT",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"\nContent: {content}")
        print(f"Expected: {expected}")
        print(f"Claude's Response:\n{response.content[0].text}")
        print("-"*60)

if __name__ == "__main__":
    demo_judge()

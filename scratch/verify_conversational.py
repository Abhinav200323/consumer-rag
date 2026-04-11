import asyncio
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.planner import run_query

async def test_conversational_output():
    query = "What can I do if I bought a defective product online and the seller refuses to refund?"
    print(f"Testing Query: {query}\n")
    
    # Run the query pipeline
    result = await run_query(query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("--- BOT RESPONSE ---")
        print(result["answer"])
        print("\n--- METADATA ---")
        print(f"Reasoning Steps count: {len(result.get('reasoning_steps', []))}")
        print(f"Citations count: {len(result.get('citations', []))}")

if __name__ == "__main__":
    asyncio.run(test_conversational_output())

import asyncio
from agent.planner import run_query

async def test():
    print("--- Test 1: Simple Greeting ---")
    res1 = await run_query("Hi there!")
    print(f"Answer: {res1['answer']}")
    print(f"Trace: {res1['trace']}")
    
    print("\n--- Test 2: Vague Query (Cross-Questioning) ---")
    res2 = await run_query("I bought a phone and it is not working")
    print(f"Answer: {res2['answer']}")
    # print(f"Trace: {res2['trace']}")

if __name__ == "__main__":
    asyncio.run(test())

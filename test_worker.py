import asyncio
import traceback
from liquid_swarm.models import TaskInput
from liquid_swarm.nodes import execute_task

async def main():
    task = TaskInput(task_id="test_task_1", query="Test the market size of AI in 2025.")
    try:
        result = await execute_task(task)
        print("STATUS:", result.status)
        print("DATA:", result.data)
    except Exception as e:
        print("EXCEPTION CAUGHT:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

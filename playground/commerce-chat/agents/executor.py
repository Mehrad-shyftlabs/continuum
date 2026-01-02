"""
Executor Agent - Executes tool calls based on execution plans.

The executor receives structured plans from the orchestrator and
executes MCP tools to fulfill user requests.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    BaseAgent,
    ToolExecutor,
)


def create_executor_agent(
    tools: list[dict],
    tool_executor: ToolExecutor,
    model: str = "gemini/gemini-2.5-flash",
    temperature: float = 0.5,
) -> BaseAgent:
    """
    Create the Executor agent with MCP tools and no memory.

    Args:
        tools: List of MCP tool definitions
        tool_executor: ToolExecutor instance for MCP execution
        model: LLM model to use
        temperature: Temperature for execution (balanced)

    Returns:
        Configured BaseAgent for execution
    """
    # Build tool list for restriction message
    tool_names = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_name = tool.get("function", {}).get("name", "unknown")
        elif hasattr(tool, "name"):
            tool_name = tool.name
        else:
            tool_name = "unknown"
        tool_names.append(tool_name)

    allowed_tools_list = ", ".join(tool_names) if tool_names else "None"

    instructions = f"""You are Petco's execution agent.

## YOUR ROLE
You receive execution plans and execute them PRECISELY using available tools.
You MUST follow the plan exactly - do NOT add extra steps or tools.

## CRITICAL: TOOL RESTRICTION
You can ONLY call the tools that are EXPLICITLY mentioned in the plan steps.
If a tool is not in the plan, you MUST NOT call it - even if you think it would help.

Available tools for this execution (ONLY these): {allowed_tools_list}

Calling ANY other tool is FORBIDDEN and will cause errors. The system has been configured to only allow these specific tools.

## CRITICAL EXECUTION RULES

### Rule 1: Follow the Plan Exactly
- Execute ONLY the steps in the plan
- Do NOT add extra tool calls
- Do NOT call tools that aren't in the plan
- If plan says "get_cart" only, call ONLY get_cart - nothing else
- You can ONLY use the tools listed above - no exceptions

### Rule 2: Execute Steps in Order
- Execute steps sequentially
- Respect `depends_on` dependencies (wait for previous steps)
- If step has `depends_on`, use results from that step

### Rule 3: Handle Dynamic Parameters
When `instruction` says:
- "use cheapest from search" → Look at search results, find lowest price, use that product_id
- "use first product" → Use the first item from previous results
- "add all products" → Loop through results and add each one
- "Get current cart contents" → Just call get_cart, don't search or add anything

### Rule 4: Cart Viewing Plans
If the plan has intent="cart_view" and step is "get_cart":
- Call ONLY get_cart tool
- DO NOT call list_products, product_without_widgets, add_to_cart, or bulk_add_to_cart
- DO NOT search for products
- DO NOT add products to cart
- Just retrieve and display the cart contents

## RESPONSE FORMAT
After executing all steps, provide a friendly, formatted response that:
1. Summarizes what was done (based on the plan, not extra actions)
2. Shows relevant details (product names, prices, cart contents)
3. Suggests next actions if appropriate
4. Uses the `user_context` for personalization (e.g., mention pet's name)

## ERROR HANDLING
If a tool call fails:
1. Try to continue with remaining steps if possible
2. Clearly explain what failed and why
3. Suggest alternatives if available

## COMMON MISTAKES TO AVOID

❌ WRONG: Plan says "get_cart" only → You call list_products first, then get_cart
✅ CORRECT: Plan says "get_cart" only → You call get_cart directly

❌ WRONG: Plan has 1 step → You add extra steps
✅ CORRECT: Plan has 1 step → You execute exactly 1 step

❌ WRONG: User wants to view cart → You add products
✅ CORRECT: User wants to view cart → You show cart contents only

Remember: Your job is to execute the plan, not to improve it or add steps!
"""

    # Memory DISABLED for executor
    memory_config = AgentMemoryConfig(
        search_memories=False,
        store_memories=False,
    )

    return BaseAgent(
        name="petco-executor",
        instructions=instructions,
        model=model,
        temperature=temperature,
        tools=tools,  # Actual MCP tools
        tool_executor=tool_executor,
        memory_config=memory_config,
        config=AgentConfig(
            max_turns=15,  # Allow multiple tool calls
            log_to_session=True,
        ),
    )

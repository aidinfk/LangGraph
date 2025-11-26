import json
from typing import List
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_community.tools import TavilySearchResults


# Create Tavily Search Tool
tavily_tool = TavilySearchResults(max_results=5)

# Function to Execute Search Queries From AnswerQuestion Tool Calls
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message = None

    for msg in reversed(state):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if not last_ai_message or not last_ai_message.tool_calls:
        return []

    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                try:
                    result = tavily_tool.invoke({"query": query})  # تغییر: اضافه کردن دیکشنری
                    query_results[query] = result
                except Exception as e:
                    query_results[query] = f"Error: {str(e)}"

            # Create a tool message with the result
            tool_messages.append(
                ToolMessage(
                    content = json.dumps(query_results),
                    tool_call_id = call_id
                )
            )

    return tool_messages

test_state = [
    HumanMessage(
        content="Write about how small business can leverage AI to grow"
    ),
    AIMessage(
        content = "",
        tool_calls = [
            {
                "name": "AnswerQuestion",
                "args": {
                'answer': '',
                'search_queries': [
                    'AI tools for small business',
                    'AI in small business marketing',
                    'AI automation for small business'
                ],
                'reflection': {
                    'missing': '',
                    'superfluous': ''
                }
                },
                "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
            }
        ]
    )
]

# Execute the tools
try:
    results = execute_tools(test_state)
    print("Raw results:", results)
    if results:
        parsed_content = json.loads(results[0].content)
        print("Parsed content:", parsed_content)
except Exception as e:
    print(f"Test error: {e}")
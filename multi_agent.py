import os
import asyncio
import re
import subprocess
from dotenv import load_dotenv
load_dotenv()

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel

# 1. Initialize Kernel and Azure OpenAI connection
kernel = Kernel()

deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

print("deployment_name:", deployment_name)  # Should NOT be None!

azure_chat_service = AzureChatCompletion(
    deployment_name=deployment_name,
    endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)

# 2. Create Persona Agents with full instructions!
business_analyst = ChatCompletionAgent(
    id="BusinessAnalyst",
    name="BusinessAnalyst",
    instructions=(
        "You are a Business Analyst who takes requirements from the user (customer) and creates a project plan "
        "for creating the requested app. You understand user requirements and create detailed documents with requirements and costing. "
        "Your documents should be usable by the SoftwareEngineer as a reference for implementation, and by the Product Owner for verification."
    ),
    service=azure_chat_service,
)
software_engineer = ChatCompletionAgent(
    id="SoftwareEngineer",
    name="SoftwareEngineer",
    instructions=(
        "You are a Software Engineer. Your goal is to create a web app using HTML and JavaScript, implementing all requirements "
        "from the Business Analyst. Deliver code to the Product Owner for review. If requirements are unclear, ask the Business Analyst for clarification."
    ),
    service=azure_chat_service,
)
product_owner = ChatCompletionAgent(
    id="ProductOwner",
    name="ProductOwner",
    instructions=(
        "You are the Product Owner. You review the software engineer's code to ensure all requirements are complete and the product meets specifications. "
        "IMPORTANT: Verify that the code is shared using the format ```html [code] ```. If all requirements are met and code is correctly formatted, reply with 'READY FOR USER APPROVAL'. "
        "Otherwise, send feedback with defect details."
    ),
    service=azure_chat_service,
)

agent_list = [business_analyst, software_engineer, product_owner]

# 3. Termination strategy
class ApprovalTerminationStrategy(TerminationStrategy):
    async def should_agent_terminate(self, agent, history):
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role == AuthorRole.USER and msg.content.strip().upper() == "APPROVED":
                    return True
        return False

# 4. Utility to extract HTML code
def extract_html_code(history):
    pattern = re.compile(r"```html(.*?)```", re.DOTALL | re.IGNORECASE)
    for msg in history:
        if hasattr(msg, "content") and msg.content:
            matches = pattern.findall(msg.content)
            if matches:
                return matches[0].strip()
    return None

# 5. Save code and push to GitHub
def save_and_push_code(html_code):
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_code)
    print("[INFO] index.html saved. Attempting to push to GitHub...")
    result = subprocess.run(["bash", "./push_to_github.sh"], capture_output=True, text=True)
    if result.returncode == 0:
        print("[INFO] index.html pushed to GitHub.")
    else:
        print("[WARNING] Git push failed:", result.stderr)

# 6. Main Multi-Agent Workflow
async def run_multi_agent(user_input: str):
    history = ChatHistory()
    history.add_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))

    group_chat = AgentGroupChat(
        agents=agent_list,
        termination_strategy=ApprovalTerminationStrategy(),
        chat_history=history
    )
    responses = []
    approval_found = False
    async for msg in group_chat.invoke():
        print(f"[{msg.role}] {getattr(msg, 'content', '')}")  # Print agent responses for debugging!
        responses.append(msg)
        # Stop as soon as the user says "APPROVED"
        if hasattr(msg, "role") and msg.role == AuthorRole.USER and msg.content.strip().upper() == "APPROVED":
            approval_found = True
            break
    if approval_found:
        html_code = extract_html_code(responses)
        if html_code:
            save_and_push_code(html_code)
        else:
            print("[WARNING] No HTML code found in chat history.")
    return responses

if __name__ == "__main__":
    user_req = input("Enter your app requirements: ")
    asyncio.run(run_multi_agent(user_req))

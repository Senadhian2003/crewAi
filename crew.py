import os
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import uuid

session_id = str(uuid.uuid4())
print(session_id)



 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
 

 
CrewAIInstrumentor().instrument(skip_dep_check=True)


model = "azure/gpt-4.1"
llm = LLM(
            model=model,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview"
        )


class FactorialGuideInput(BaseModel):
    question: str = Field(
        ...,
        description="Clarifying question that the coder wants the guardrails tool to address.",
    )


class FactorialGuideTool(BaseTool):
    """Dummy tool that provides guardrails for factorial implementations."""

    name: str = "factorial_reference_tool"
    description: str = (
        "Consult this before writing factorial code. "
        "Input: factorial_reference_tool(question='your question here'). "
        "It returns guardrails, error handling expectations, and template ideas."
    )
    args_schema: type[BaseModel] = FactorialGuideInput

    def _run(self, question: str) -> str:
        return (
            "Factorial guardrails:\n"
            "- Accept integers only and raise ValueError for negatives.\n"
            "- Provide recursion with explicit base cases of 0! = 1 and 1! = 1.\n"
            "- Include docstrings, examples, and warnings for large inputs.\n"
            "- Keep the function pure and deterministic.\n"
            f"Question received: {question}"
        )


class RecursionInsightInput(BaseModel):
    question: str = Field(
        ...,
        description="Question describing what needs to be clarified about recursion.",
    )


class RecursionInsightTool(BaseTool):
    """Dummy tool that returns narrative-friendly recursion explanations."""

    name: str = "recursion_insight_tool"
    description: str = (
        "Use to gather teaching metaphors and stack-walk explanations for recursion. "
        "Input: recursion_insight_tool(question='your question here')."
    )
    args_schema: type[BaseModel] = RecursionInsightInput

    def _run(self, question: str) -> str:
        return (
            "Recursion insight:\n"
            "- Highlight difference between base case and recursive step.\n"
            "- Describe call stack frames as pending multiplications for factorial.\n"
            "- Emphasize unwinding phase returning accumulated values.\n"
            "- Provide analogies (nesting dolls, stack of plates) to explain flow.\n"
            f"Question received: {question}"
        )


coder_tools = [FactorialGuideTool()]
explainer_tools = [RecursionInsightTool()]

 
# Define your agents with roles and goals
coder = Agent(
    role='Python Developer',
    goal='Write clean, efficient, and well-documented Python code with proper error handling',
    backstory='You are an experienced Python developer specializing in algorithms and data structures. You write production-ready code with comprehensive error handling and clear documentation.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=10,
    tools=coder_tools,
)

explainer = Agent(
    role='Technical Educator',
    goal='Provide clear, detailed explanations of programming concepts and how code works',
    backstory='You are a patient and thorough technical educator who excels at breaking down complex programming concepts into understandable explanations. You use examples and analogies to help others learn.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=10,
    tools=explainer_tools,
)
 
# Create tasks for your agents
question1 = (
    "Consult factorial_reference_tool with a clarifying question, summarize the guardrails, "
    "then craft a recursive factorial(number: int) implementation with docstring, "
    "negative-number error handling, and a worked example. Return a structured plan "
    "plus the final code so Task 2 can explain it."
)
task1 = Task(
    description=question1,
    expected_output=(
        "1) Tool citation + summary\n"
        '2) Step-by-step plan for building factorial(number: int)\n'
        "3) Final Python function string and usage example"
    ),
    agent=coder,
    async_execution=False,
)

question2 = (
    "Leverage recursion_insight_tool to gather metaphors. Review Task 1 output, "
    "explain how the factorial implementation works, detail call stack behavior, "
    "and produce the final answer for the user. Explicitly mention how the tool "
    "informed the explanation."
)
task2 = Task(
    description=question2,
    expected_output=(
        "Educational narrative describing recursion, base case, and stack unwinding, "
        "plus a concise final response ready for the user"
    ),
    agent=explainer,
    context=[task1],
    async_execution=False,
)


 

 
# Instantiate your crew
crew = Crew(
    agents=[coder, explainer],
    tasks=[task1, task2],
    verbose=True,
    tracing=False,
    output_log_file=False,
    planning=True,
    process=Process.sequential,
)

trace_input = {
                "task1": question1,
                "task2": question2
            }
 
with langfuse.start_as_current_observation(as_type="span", name="crewai-index-trace") as span:
    with propagate_attributes(
        session_id=session_id,
        version="1.0.0",
        ):
        result = crew.kickoff()
        print(result)
        span.update_trace(
            input=trace_input,
            output=str(result)[:1000] if result else "No result",
        )
    
langfuse.flush()
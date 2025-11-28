import os
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, LLM, Process
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

 
# Define your agents with roles and goals
coder = Agent(
    role='Python Developer',
    goal='Write clean, efficient, and well-documented Python code with proper error handling',
    backstory='You are an experienced Python developer specializing in algorithms and data structures. You write production-ready code with comprehensive error handling and clear documentation.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=10,
)

explainer = Agent(
    role='Technical Educator',
    goal='Provide clear, detailed explanations of programming concepts and how code works',
    backstory='You are a patient and thorough technical educator who excels at breaking down complex programming concepts into understandable explanations. You use examples and analogies to help others learn.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=10,
)
 
# Create tasks for your agents
question1 = "Write a Python function that calculates the factorial of a number using recursion. Include proper error handling for negative numbers."
task1 = Task(
    description=question1,
    expected_output="A complete Python function with recursion and error handling",
    agent=coder
)

question2 = "Explain how recursion works in programming, using the factorial calculation as an example. Describe the call stack and base case."
task2 = Task(
    description=question2,
    expected_output="A detailed explanation of recursion concepts with examples",
    agent=explainer,
  
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
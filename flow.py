from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

os.environ["SERPER_API_KEY"] = "bcdfd0e7264879b21bc9615724ec4cd9a3a54ba6"
os.environ["WATSONX_APIKEY"] = "U1uZYpjUXPCfhd2_30EMPqRwFZCqnffq3rCBGakUJp8X"

search = SerperDevTool()

parameters = {"decoding_method": "greedy", "max_new_tokens": 2048, "temperature": 0.1}

llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="0d182389-14e9-40b0-807e-392bdae7c0d1",
    params=parameters,
)

function_llm = WatsonxLLM(
    model_id="ibm-mistralai/merlinite-7b",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="0d182389-14e9-40b0-807e-392bdae7c0d1",
    params=parameters,
)

researcher = Agent(
    llm=llm,
    role="Senior AI Researcher",
    goal="Find promising research in the field of quantum computing.",
    backstory="You are a veteran quantum computing researcher with a background in modern physics.",
    allow_delegation=False,
    function_calling_llm=function_llm,
    tools=[search],
    verbose=True,
)

writer = Agent(
    llm=llm,
    role="Senior Speech Writer",
    goal="Write engaging and witty keynote speeches from provided research.",
    backstory="You are a deeply experienced speech writer who has written countless keynote presentations for executives.",
    allow_delegation=False,
    verbose=True,
)

task1 = Task(
    description="Search the internet and find 5 examples of promising AI research.",
    expected_output="A detailed bullet point summary on each of the topics. Each bullet point should cover the topic, background and why the innovation is useful.",
    agent=researcher,
    output_file="task1_output.txt",
)

task2 = Task(
    description="Write an engaging keynote speech on quantum computing.",
    expected_output="A detailed keynote speech with an intro, body and conclusion.",
    agent=writer,
    output_file="task2_output.txt",
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)

res = crew.kickoff()
print(res)

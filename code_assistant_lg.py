from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
import os
import subprocess
import pkg_resources

os.environ["AZURE_OPENAI_ENDPOINT"] = "azure deployment end point"
os.environ["AZURE_OPENAI_API_KEY"] = "provide-your-key"

llm = AzureChatOpenAI(
    azure_deployment="gpt-4",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in python language. \n 
             Structure your answer with list of imports \n
             and the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

code_gen_prompt_package_install = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in python language. \n 
               you need to provide name of package to be installed 
               output expected:
               display only name of the packagename, not pip install packagename""",
        ),
        ("placeholder", "{messages}"),
    ]
)


class Code(BaseModel):
    """Schema for code solutions"""

    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""

    return solution["parsed"]


question = "selenium webdriver python code to launch a webdriver and close it"
code_gen_chain = code_gen_prompt | llm.with_structured_output(Code, include_raw=True) | parse_output
code_gen_chain_package_install = code_gen_prompt_package_install | llm

# solution = code_gen_chain.invoke(
#     {"messages": [("user", question)]}
# )
# print(solution)


from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int
    package_failed: str


### Parameter

# Max tries
max_iterations = 3


def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a imports, and code block:",
            )
        ]

    # Solution
    code_solution = code_gen_chain.invoke(
        {"messages": messages}
    )
    messages += [
        (
            "assistant",
            f"Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "package_failed": "yes"
        }

    # Check execution
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "package_failed": "no"
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
        "package_failed": "no"
    }


def check_package(state: GraphState):
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]
    package_failed = state["package_failed"]

    if package_failed:
        package_name = code_gen_chain_package_install.invoke({"messages": state["messages"]})
        print("---Checking package failure---")

        try:
            # Check if the package is already installed
            pkg_resources.get_distribution(package_name.content)
            print(f"{package_name} is already installed.")
        except Exception as e:
            # If not installed, install the package
            print(f"Installing {package_name.content}...")
            subprocess.run(['pip3', 'install', package_name.content], check=True)
            print(f"{package_name.content} installed successfully.")
            messages += [
                (
                    "assistant",
                    f"{package_name.content} installed successfully.",
                )
            ]

    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
        "package_failed": "no"
    }


### Edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("check_package", check_package)

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
workflow.add_edge("check_code", "check_package")
workflow.add_conditional_edges(
    "check_package",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

app = workflow.compile()

solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": "", "package_failed": ""})
from pprint import pprint

pprint(solution)

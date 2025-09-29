import smolagents
import click
from typing import Optional, List, Tuple
from smolagents import LiteLLMModel, CodeAgent
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
    tool,
    PromptTemplates,
    EMPTY_PROMPT_TEMPLATES,
    PlanningPromptTemplate,
)

from aim import Repo
from scipy.signal import savgol_filter
from aim import Run
from smolagents.models import OpenAIServerModel
import os

system_prompt = """
## Instructions
You are acting as a expert data analyst and graph visualization specialist.
Given a AimStack Repository.

## Analytics Steps
1. Get the runs and their metrics from the Aim repository using the tool get_aim_experiment_data_from_hash_id, parsing the repository path and experiment hash from the user query.
2. For each run, get the specified metric asked for
3. Generate a visualizations plot with time/steps on the x-axis and the value on the y-axis (often loss-value).


## Data information
# Tuple[List[int], List[float]]: A tuple containing two lists - x-values and y-values.

## User Question:
{query}
"""

def smoothening(vector, smooth):
    if smooth is None:
        return vector
    alg, window = smooth[:2]
    if alg == "mean":
        return [sum(vector[i:i+window])/window for i in range(len(vector)-window+1)]
    elif alg == "median":
        return [sorted(vector[i:i+window])[window//2] for i in range(len(vector)-window+1)]
    elif alg == "exponential":
        alpha = 2/(window+1)
        result = [vector[0]]
        for i in range(1, len(vector)):
            result.append(alpha*vector[i] + (1-alpha)*result[-1])
        return result
    elif alg == "savitzky-golay":
        return savgol_filter(vector, window, smooth[2])
    else:
        raise ValueError(f"unknown smoothing algorithm {alg}")
    

@tool
def get_aim_experiment_data_from_hash_id(
    repo: str, 
    exp_hash: str,
    metric: str="train/loss", 
    min_x: int=0, 
    max_x: int=1_000_000, 
    x_offset: int = 0, 
    scale: float = 1.0, 
    smooth: Tuple[str, int, int] = ["savitzky-golay", 64, 2],
    _xtimeoffset: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
    """
    Fetches experiment data from Aim repository based on experiment hash and metric name.
    Args:
        repo (str): Path to the Aim repository.
        exp_hash (str): Experiment hash identifier.
        metric (str): Name of the metric to fetch.
        min_x (int): Minimum x-value (index) to consider.
        max_x (int): Maximum x-value (index) to consider.
        x_offset (int, optional): Offset to apply to x-values. Defaults to 0
        scale (float, optional): Scaling factor for metric values. Defaults to 1.0.
        smooth (Tuple[str, int, int]): Smoothing algorithm and parameters. Defaults to ["savitzky-golay", 64, 2].
        _xtimeoffset (Optional[int], optional): Time offset for x-values. Defaults to None.
    Returns:
        Tuple[List[int], List[float]]: A tuple containing two lists - x-values and y-values.
    """
    run = Run(run_hash=exp_hash, repo=repo, read_only=True)
    if run is None:
        raise ValueError(f"Experiment with hash {exp_hash} not found in repo {repo}")

    data = None  # (indices, values)
    for seq in run.metrics():
        print(f"Found metric: {seq.name}")
        if seq.name == metric:
            print(  f"Fetching metric {metric} for {exp_hash}")
            raw_data = [val/scale for _, (val, _, _) in seq.data.items()]
            max_x = min(max_x, len(raw_data))
            raw_data = raw_data[min_x:max_x]
            # raw_data = smoothening(raw_data, smooth)
            offset = x_offset
            if _xtimeoffset is not None:
                indices = [t for _, (_, _, t) in seq.data.items()]
                indices = [(t-indices[0]+_xtimeoffset) for t in indices]
                indices = indices[min_x:max_x] # get indices within range
                _xtimeoffset = 2*indices[-1]-indices[-2]
            else:
                indices = list(range(1+offset, len(raw_data)+1+offset))
            
            data = (indices, raw_data)
            break
        else:
            print(f"metric {metric} not found for {exp_hash} - skipping")
            continue
    return data

@click.group()
def _agent():
    pass
@_agent.command()
@click.argument("query", type=str)
def agent(query: str):
    base_url = os.environ.get("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")
    model = os.environ.get("OPENAI_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")
    model = OpenAIServerModel(model_id=model, api_key=api_key, api_base=base_url)
    agent = CodeAgent(
        tools=[get_aim_experiment_data_from_hash_id],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=[
            "pandas", "numpy", "datetime", "matplotlib", "sklearn", "matplotlib.pyplot",
        ],
        instructions=system_prompt,
    )
    response = agent.run(query)
    click.echo(response)

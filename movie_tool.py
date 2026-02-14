from openai import pydantic_function_tool
from pydantic import BaseModel, Field
import subprocess
import sys


def get_dataframe_schema(df):
    schema = f"Columns: {df.columns.tolist()}\n"
    schema += f"Data types:\n{df.dtypes.to_string()}\n"
    schema += f"Shape: {df.shape}\n"
    schema += f"Sample:\n{df.head(3).to_string()}"
    return schema


class QueryMovieDB(BaseModel):
    """Query the movie database using Python code."""
    code: str = Field(description="Python code to execute. Must use print() to output results.")


def get_tools(filtered_df):
    schema = get_dataframe_schema(filtered_df)
    return [pydantic_function_tool(
        QueryMovieDB,
        description=f"Execute Python code to query the movie database. The DataFrame `df` is pre-loaded. Always use print() to output results.\n\nSchema:\n{schema}"
    )]


def query_movie_db(code, filtered_df):
    filtered_df.to_csv('temp_data.csv', index=False)

    full_code = f"""import pandas as pd
import numpy as np

df = pd.read_csv('temp_data.csv')

{code}"""

    with open("generated_code.py", "w") as f:
        f.write(full_code)

    result = subprocess.run(
        [sys.executable, "generated_code.py"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode != 0:
        return result.stderr
    if not result.stdout.strip():
        return "No output. Did you forget to use print()?"
    return result.stdout

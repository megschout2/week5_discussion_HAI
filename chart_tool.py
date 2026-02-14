from openai import pydantic_function_tool
from pydantic import BaseModel, Field
import altair as alt
import json


class CreateChart(BaseModel):
    """Create a chart visualization using a Vega-Lite specification."""
    vega_lite_spec: str = Field(description="A complete Vega-Lite JSON specification string, including inline data under 'data.values'.")


def get_chart_tool():
    return pydantic_function_tool(
        CreateChart,
        description="Create a visualization by providing a Vega-Lite JSON specification. The data should be included inline in the spec under the 'data.values' field. Use this when the user asks for a visualization, chart, plot, or graph."
    )


def validate_chart(vega_lite_spec):
    try:
        spec = json.loads(vega_lite_spec)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    try:
        alt.Chart.from_dict(spec)
        return spec, "Valid Vega-Lite specification."
    except Exception as e:
        return None, f"Invalid Vega-Lite specification: {e}"

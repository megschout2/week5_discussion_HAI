import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import Optional
from movie_tool import get_tools, query_movie_db
from chart_tool import get_chart_tool, validate_chart


# ── State ──

DEFAULT_STATE = {
    "agent_phase": "idle",
    "agent_events": [],
    "agent_messages": [],
    "agent_tools": [],
    "agent_df": None,
    "agent_chart_specs": [],
    "agent_pending_message": None,
    "agent_pending_final": None,
}

def get_state(key):
    return st.session_state.get(key, DEFAULT_STATE[key])

def set_state(key, value):
    st.session_state[key] = value

def restart_agent(user_question, filtered_df, show_chart=False):
    set_state("agent_phase", "thinking")
    set_state("agent_events", [])
    set_state("agent_chart_specs", [])
    set_state("agent_pending_message", None)

    tools = get_tools(filtered_df)
    system_content = "You are a data analyst with access to a tool that executes Python code on a movie database."

    if show_chart:
        tools.append(get_chart_tool())
        system_content += " After computing the data, create a chart using a Vega-Lite specification."

    set_state("agent_messages", [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
    ])
    set_state("agent_tools", tools)
    set_state("agent_df", filtered_df)


# ── Logic ──

def run_step(client):
    phase = get_state("agent_phase")
    messages = get_state("agent_messages")

    # Safety: if messages already contains an assistant message that includes
    # tool_calls but the corresponding tool messages are not present yet,
    # move to awaiting_approval instead of calling the API (prevents 400 error).
    pending_assistant = None
    tool_call_ids_seen = set()
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id"):
            tool_call_ids_seen.add(m.get("tool_call_id"))
    for m in messages:
        # messages from the OpenAI SDK may be objects with attribute access
        tcalls = None
        if isinstance(m, dict):
            tcalls = m.get("tool_calls")
        else:
            tcalls = getattr(m, "tool_calls", None)
        if tcalls:
            # check if all tool_call_ids have corresponding tool messages
            missing = []
            for tc in tcalls:
                if tc.id not in tool_call_ids_seen:
                    missing.append(tc.id)
            if missing:
                # set pending and switch to awaiting_approval so user can act
                set_state("agent_pending_message", m)
                set_state("agent_phase", "awaiting_approval")
                return

    if phase == "thinking":
        class Reasoning(BaseModel):
            reason: str = Field(description="Your reasoning about what you know so far and what to do next")
            use_tool: bool = Field(description="True if you need to run code or create a chart, False if you can give the final answer")
            answer: Optional[str] = Field(default=None, description="Your final answer in one short paragraph. Only provide when use_tool is False.")

        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=Reasoning,
        )
        reasoning = response.choices[0].message.parsed
        messages.append({"role": "assistant", "content": reasoning.reason})

        if reasoning.use_tool:
            get_state("agent_events").append({"type": "thought", "thought": reasoning.reason})
            set_state("agent_phase", "acting")
        else:
                # Instead of finishing immediately, send a proposed final answer
                # to the user for review. This introduces a richer human-in-the-loop
                # where the user can accept, request more analysis, or edit the answer.
                get_state("agent_events").append({"type": "thought", "thought": reasoning.reason})
                set_state("agent_pending_final", {"thought": reasoning.reason, "answer": reasoning.answer})
                set_state("agent_phase", "awaiting_final_approval")

    elif phase == "acting":
        tools = get_state("agent_tools")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools, parallel_tool_calls=False,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            set_state("agent_phase", "done")
            return

        set_state("agent_pending_message", msg)
        set_state("agent_phase", "awaiting_approval")

def execute_pending_tools():
    # Default behavior: execute all tool calls as-is
    return execute_pending_tools_with_actions(None)


def execute_pending_tools_with_actions(tool_actions):
    messages = get_state("agent_messages")
    df = get_state("agent_df")
    pending_msg = get_state("agent_pending_message")

    # We'll collect tool response messages and only append the assistant message
    # with its tool outputs after all tool responses are ready. This prevents
    # sending an assistant message with `tool_calls` without corresponding
    # tool messages (which triggers the OpenAI 400 error).
    tool_messages = []

    # Build a map for quick lookup if user provided actions/edits
    action_map = {}
    if tool_actions:
        for ta in tool_actions:
            action_map[str(ta["id"])]=ta

    for tc in pending_msg.tool_calls:
        ta = action_map.get(str(tc.id)) if action_map else None

        # Skip: record rejected event and a tool message noting skip
        if ta and ta.get("action") == "Skip":
            get_state("agent_events").append({"type": "rejected", "name": tc.function.name, "feedback": "User skipped this tool."})
            tool_messages.append({"role": "tool", "content": f"Skipped {tc.function.name}", "tool_call_id": tc.id})
            continue

        # (Removed 'Ask for Clarification' handling.)

        # Edit and Run
        if ta and ta.get("action") == "Edit and Run":
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            if tc.function.name == "QueryMovieDB":
                code = ta.get("edited", args.get("code",""))
                result = query_movie_db(code, df)
                get_state("agent_events").append({"type": "action", "name": tc.function.name, "code": code, "result": result,})
                tool_messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            elif tc.function.name == "CreateChart":
                spec_str = ta.get("edited", args.get("vega_lite_spec",""))
                spec, result = validate_chart(spec_str)
                if spec:
                    get_state("agent_chart_specs").append(spec)
                get_state("agent_events").append({"type": "chart", "name": tc.function.name, "spec_str": spec_str, "result": result,})
                tool_messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            continue

        # Default run-as-is path
        args = json.loads(tc.function.arguments)
        if tc.function.name == "QueryMovieDB":
            result = query_movie_db(args.get("code",""), df)
            get_state("agent_events").append({"type": "action", "name": tc.function.name, "code": args.get("code",""), "result": result,})
            tool_messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
        elif tc.function.name == "CreateChart":
            spec, result = validate_chart(args.get("vega_lite_spec",""))
            if spec:
                get_state("agent_chart_specs").append(spec)
            get_state("agent_events").append({"type": "chart", "name": tc.function.name, "spec_str": args.get("vega_lite_spec",""), "result": result,})
            tool_messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})

    # All tool messages collected — append the assistant message then the tool messages
    messages.append(pending_msg)
    for tm in tool_messages:
        messages.append(tm)

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")

def reject_pending_tools(feedback):
    messages = get_state("agent_messages")
    pending_msg = get_state("agent_pending_message")

    rejection_msg = "User rejected this action."
    if feedback:
        rejection_msg += f" User feedback: {feedback}"
    else:
        rejection_msg += " Try a different approach."

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        get_state("agent_events").append({
            "type": "rejected", "name": tc.function.name,
            "feedback": feedback,
        })
        messages.append({"role": "tool", "content": rejection_msg, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")


# ── Rendering ──

def render_events():
    for event in get_state("agent_events"):
        if event["type"] == "thought":
            st.markdown(f"**Thought:** {event['thought']}")
        elif event["type"] == "action":
            st.markdown(f"**Action:** `{event['name']}`")
            code_text = event.get("code")
            if code_text:
                st.code(code_text, language="python")
            st.markdown("**Observation:**")
            st.code(event.get("result", "(no result)"), language="text")
            st.divider()
        elif event["type"] == "chart":
            st.markdown(f"**Action:** `{event['name']}`")
            spec_text = event.get("spec_str")
            if spec_text:
                st.code(spec_text, language="json")
            st.markdown("**Observation:**")
            st.code(event.get("result", "(no result)"), language="text")
            st.divider()
        elif event["type"] == "rejected":
            st.markdown(f"**Rejected:** `{event['name']}`")
            if event.get("feedback"):
                st.text(f"Feedback: {event['feedback']}")
            st.divider()
        elif event["type"] == "answer":
            st.markdown(f"**Thought:** {event['thought']}")

def render_pending_approval():
    pending = get_state("agent_pending_message")
    if not pending:
        st.error("No pending tool call to approve.")
        return {"submitted": False}

    st.warning("The agent wants to perform the following action. You can run each tool as-is, edit its input, or skip it.")

    tool_actions = []
    for i, tc in enumerate(pending.tool_calls):
        args = json.loads(tc.function.arguments)
        st.markdown(f"**Tool {i+1}:** `{tc.function.name}` (id: {tc.id})")

        if tc.function.name == "QueryMovieDB":
            default_text = args.get("code", "")
            st.markdown("Edit Python code that will run against the preloaded DataFrame `df`.")
            edited = st.text_area(f"Code for {tc.id}", value=default_text, key=f"code_{tc.id}")
        elif tc.function.name == "CreateChart":
            default_text = args.get("vega_lite_spec", "")
            st.markdown("Edit Vega-Lite JSON spec (must include data.values if editing).")
            edited = st.text_area(f"Spec for {tc.id}", value=default_text, key=f"spec_{tc.id}")
        else:
            edited = tc.function.arguments

        action = st.selectbox(
            f"Action for {tc.function.name} ({tc.id})",
            options=["Run as-is", "Edit and Run", "Skip"],
            key=f"action_{tc.id}",
        )

        st.divider()
        tool_actions.append({"id": tc.id, "name": tc.function.name, "action": action, "edited": edited})

    submitted = st.button("Submit Actions", type="primary", use_container_width=True)
    return {"submitted": submitted, "tool_actions": tool_actions}


def render_pending_final_approval():
    pending = get_state("agent_pending_final")
    if not pending:
        st.error("No pending final answer to review.")
        return {"accept": False, "accept_edited": False, "accept_export": False, "accept_edited_export": False, "request_more": False, "edited_text": "", "confidence": 0, "export": False}
    st.warning("The agent proposes the following final answer. You can accept it, request more analysis, or edit the answer:")
    st.markdown("**Proposed Answer (short):**")
    st.write(pending.get("answer"))
    with st.expander("Agent reasoning (summary)", expanded=False):
        st.write(pending.get("thought"))

    edit = st.text_area("Edit proposed answer (optional)", value=pending.get("answer"), key="final_edit")

    col1, col2, col3 = st.columns(3)
    with col1:
        accept = st.button("Accept Answer", use_container_width=True, type="primary")
    with col2:
        accept_edited = st.button("Accept Edited Answer", use_container_width=True)
    with col3:
        request_more = st.button("Request More Analysis", use_container_width=True)

    return {
        "accept": accept,
        "accept_edited": accept_edited,
        "request_more": request_more,
        "edited_text": edit,
    }

def render_pending_feedback():
    feedback = st.text_input(
        "Why are you rejecting? Tell the agent what to do instead:",
        key="reject_feedback",
    )
    submitted = st.button("Submit Rejection", use_container_width=True)
    return submitted, feedback

def render_panel():
    st.subheader("Analysis Results")
    container = st.container(height=600)
    actions = {}
    with container:
        phase = get_state("agent_phase")

        if phase == "idle":
            st.info("Enter a question and click 'Analyze' to see results.")

        elif phase in ("thinking", "acting"):
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            st.spinner("Agent is thinking...")

        elif phase == "awaiting_approval":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            # render_pending_approval now returns a dict with keys 'submitted' and 'tool_actions'
            pending_actions = render_pending_approval()
            actions = pending_actions

        elif phase == "awaiting_final_approval":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            final_actions = render_pending_final_approval()
            actions = {"final_actions": final_actions}

        elif phase == "awaiting_feedback":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            submitted, feedback = render_pending_feedback()
            actions = {"submitted": submitted, "feedback": feedback}

        elif phase == "done":
            with st.expander("Agent Reasoning Trace", expanded=False):
                render_events()
            events = get_state("agent_events")
            if events and events[-1].get("answer"):
                st.write("**Answer:**")
                st.write(events[-1]["answer"])
            for spec in get_state("agent_chart_specs"):
                st.vega_lite_chart(spec, use_container_width=True)

    return actions


# ── Lifecycle ──

def agent_panel(client, analyze_button, user_question, filtered_df, show_chart=False):
    # Phases: idle -> thinking <-> acting -> awaiting_approval -> thinking ... -> done
    #                                     -> awaiting_feedback -> thinking ... -> done
    if analyze_button and user_question:
        restart_agent(user_question, filtered_df, show_chart)

    actions = render_panel()

    phase = get_state("agent_phase")
    if phase in ("thinking", "acting"):
        run_step(client)
        st.rerun()
    elif phase == "awaiting_approval":
        # actions is now a dict: {submitted: bool, tool_actions: [...]}
        if actions.get("submitted"):
            tool_actions = actions.get("tool_actions", [])
            # Normalize tool_actions to mapping of ids -> action dict
            ta_map = []
            for ta in tool_actions:
                ta_map.append({"id": ta.get("id"), "action": ta.get("action"), "edited": ta.get("edited")})
            execute_pending_tools_with_actions(ta_map)
            st.rerun()
    elif phase == "awaiting_feedback" and actions.get("submitted"):
        reject_pending_tools(actions.get("feedback", ""))
        st.rerun()
    elif phase == "awaiting_final_approval":
        final_actions = actions.get("final_actions", {})
        if final_actions.get("accept"):
            pending = get_state("agent_pending_final")
            get_state("agent_events").append({"type": "answer", "thought": pending.get("thought"), "answer": pending.get("answer")})
            set_state("agent_pending_final", None)
            set_state("agent_phase", "done")
            st.rerun()
        elif final_actions.get("accept_edited"):
            edited = final_actions.get("edited_text")
            pending = get_state("agent_pending_final")
            get_state("agent_events").append({"type": "answer", "thought": pending.get("thought"), "answer": edited})
            set_state("agent_pending_final", None)
            set_state("agent_phase", "done")
            st.rerun()
        elif final_actions.get("request_more"):
            feedback = st.session_state.get("final_edit", "")
            msg = f"User requests more analysis. Guidance: {feedback}" if feedback else "User requests more analysis."
            messages = get_state("agent_messages")
            messages.append({"role": "user", "content": msg})
            set_state("agent_messages", messages)
            set_state("agent_pending_final", None)
            set_state("agent_phase", "thinking")
            st.rerun()

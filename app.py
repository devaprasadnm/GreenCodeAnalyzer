import streamlit as st
from analyzer.static_analyzer import StaticAnalyzer
from analyzer.dynamic_profiler import DynamicProfiler
from analyzer.prompt_builder import build_prompt
from analyzer.gemini_client import GeminiClient

# Instantiate the client early to get model list
client = GeminiClient()
try:
    available_models = client.get_available_models()
    # Choose a sensible default: pick the model with the largest output_token_limit
    default_index = 0
    best_limit = -1
    if available_models:
        for i, model in enumerate(available_models):
            try:
                meta = client.client.models.get(model=model)
                out_limit = getattr(meta, "output_token_limit", None)
                if isinstance(out_limit, int) and out_limit > best_limit:
                    best_limit = out_limit
                    default_index = i
            except Exception:
                # ignore models we can't fetch metadata for
                continue
except Exception as e:
    st.error(f"Could not fetch available models: {e}")
    available_models = []
    default_index = 0

st.title("Green IT Code Analyzer")

language = st.selectbox("Code Language", ["python"])  # initially only Python
code = st.text_area("Paste your code here", height=300)

# Gemini generation controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    model_name = st.selectbox(
        "Model",
        options=available_models,
        format_func=lambda name: name.split('/')[-1],  # show "gemini-pro"
        index=default_index
    )
    # Try to fetch model metadata to determine token limits
    model_meta = None
    try:
        # client.client.models.get accepts the model name as the 'model' kwarg
        model_meta = client.client.models.get(model=model_name)
    except Exception:
        model_meta = None
    # Show model metadata to the user when available
    if model_meta is not None:
        display = getattr(model_meta, "display_name", None) or model_name.split('/')[-1]
        in_limit = getattr(model_meta, "input_token_limit", None)
        out_limit = getattr(model_meta, "output_token_limit", None)
        st.markdown(f"**Model:** {display}  ")
        st.markdown(f"**Input token limit:** {in_limit} — **Output token limit:** {out_limit}")
with col2:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
with col3:
    # Use model's output_token_limit when available to set a safer max
    default_max = 512
    max_allowed = 4096
    if model_meta is not None:
        out_limit = getattr(model_meta, "output_token_limit", None)
        if isinstance(out_limit, int) and out_limit > 0:
            max_allowed = max(64, out_limit)
            default_max = min(default_max, max_allowed)

    max_tokens = st.number_input("Max tokens", min_value=64, max_value=max_allowed, value=default_max)
    if model_meta is not None:
        out_limit = getattr(model_meta, "output_token_limit", None)
        if isinstance(out_limit, int) and out_limit > 0 and max_tokens > out_limit:
            st.warning(f"Selected max_tokens ({max_tokens}) exceeds model's output limit ({out_limit}). The response may be truncated.")

if st.button("Analyze"):
    if not code or not code.strip():
        st.warning("Please paste some code to analyze.")
        st.stop()
    
    with st.spinner("Running static analysis…"):
        sa = StaticAnalyzer(code, language)
        static_report = sa.analyze()

    with st.spinner("Running dynamic profiling (on sample / sandbox)…"):
        # For safety, you may run only a user-supplied function, or run sandboxed/dry-run
        # Here for demo, we assume the code defines a function `main()` with no args
        env = {}
        try:
            exec(code, env)
            func = env.get("main")
            if func is None:
                st.error("Please define a `main()` function to run profiling.")
                st.stop()

            dp = DynamicProfiler(func)
            dyn_report, _ = dp.profile()

        except Exception as e:
            st.error(f"Error executing code for profiling: {e}")
            st.stop()

    prompt = build_prompt(language, code, static_report, dyn_report)
    suggestion = client.generate_suggestions(
        prompt,
        model=model_name,
        temperature=temperature,
        max_tokens=int(max_tokens),
    )

    st.subheader("Static Analysis Issues")
    st.write(static_report["issues"])

    st.subheader("Dynamic Profiling Summary")
    st.write({
        "CPU time (s)": dyn_report["cpu_time"],
        "Memory change (bytes)": dyn_report["mem_used"],
        "Peak memory (bytes)": dyn_report["peak_tracemalloc"],
    })
    st.code(dyn_report["profile_stats"], language="text")

    st.subheader("Optimization Suggestions (via Gemini)")
    st.write(suggestion)

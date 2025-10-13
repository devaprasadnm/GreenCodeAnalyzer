# ---------------------------
# CARBON EMISSIONS ESTIMATION
# ---------------------------
def estimate_carbon_emissions(runtime_seconds, cpu_power_w=65, cpu_util=0.8, mem_gb=0.2, mem_power_per_gb=3, carbon_intensity=0.475):
    # runtime_seconds: measured execution time in seconds
    # cpu_power_w: typical CPU power in watts
    # cpu_util: average CPU utilization (0-1)
    # mem_gb: average memory usage in GB
    # mem_power_per_gb: DRAM power per GB in watts
    # carbon_intensity: kgCO2e per kWh
    time_h = runtime_seconds / 3600 if runtime_seconds else 0
    energy_cpu = cpu_power_w * cpu_util * time_h
    energy_mem = mem_power_per_gb * mem_gb * time_h
    energy_total_wh = energy_cpu + energy_mem
    energy_total_kwh = energy_total_wh / 1000
    carbon_kg = energy_total_kwh * carbon_intensity
    return carbon_kg
import streamlit as st
import sqlite3
import tempfile
import os
import time
import subprocess
import sys
import uuid
import ast
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List
import zipfile
import shutil

# Optional dependency: psutil
try:
    import psutil
except Exception:
    psutil = None

# ---------------------------
# CONFIG
# ---------------------------
DB_PATH = "green_it_analyzer.db"
MAX_RUN_SECONDS = 10  # max time to allow dynamic execution
SAMPLING_INTERVAL = 0.05  # seconds for psutil sampling

SUPPORTED_LANGS = ["python", "javascript", "java"]

# ---------------------------
# DATABASE
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            filename TEXT,
            language TEXT,
            static_issues TEXT,
            runtime_seconds REAL,
            max_rss_kb INTEGER,
            avg_cpu_percent REAL,
            success INTEGER,
            notes TEXT
        )"""
    )
    conn.commit()
    conn.close()

def save_analysis(record: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analyses (id,timestamp,filename,language,static_issues,
                                 runtime_seconds,max_rss_kb,avg_cpu_percent,success,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            record.get("id"),
            record.get("timestamp"),
            record.get("filename"),
            record.get("language"),
            "\n".join(record.get("static_issues", [])),
            record.get("runtime_seconds"),
            record.get("max_rss_kb"),
            record.get("avg_cpu_percent"),
            1 if record.get("success") else 0,
            record.get("notes", ""),
        ),
    )
    conn.commit()
    conn.close()

# ---------------------------
# STATIC ANALYSIS FUNCTIONS
# ---------------------------

def analyze_python_static(code: str) -> List[str]:
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")
        return issues

    # Heuristics
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0
            self.max_loop_depth = 0
            self.recursive = False
            self.func_defs = {}

        def visit_FunctionDef(self, node):
            self.func_defs[node.name] = node
            self.generic_visit(node)

        def visit_For(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node):
            self.loop_depth += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node):
            # recursion detection (simple)
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # if inside same name? simplistic
                # not full-proof; catch obvious recursion
                # We'll detect later by scanning function bodies for calls to same name.
            self.generic_visit(node)

    v = Visitor()
    v.visit(tree)

    # Detect large comprehensions constructing lists
    if re.search(r"\[.*for .* in .*?\]", code, flags=re.S):
        issues.append("List comprehension / building large lists: consider using generators or iterators to avoid high peak memory.")

    # Detect nested loops
    if v.max_loop_depth >= 2:
        issues.append(f"Nested loops detected (max depth {v.max_loop_depth}). Consider if loops can be flattened, vectorized (numpy/pandas), or broken into streaming processing.")

    # Detect use of global mutable default args
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.defaults:
                if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                    issues.append(f"Function `{node.name}` has mutable default argument; replace with None and set within function to avoid unexpected memory growth.")

    # Detect naive file reads
    if re.search(r"open\([^,]*\)\.read\(", code) or re.search(r"open\([^,]*\)\.read\(", code):
        issues.append("Calling `.read()` on a file can load entire file into memory; use streaming reads or iterate over file lines.")

    # Detect usage of pandas read_* without chunksize
    if re.search(r"pd\.read_csv\(", code):
        issues.append("Using pandas.read_csv without `chunksize` can blow memory for large files. Consider using `chunksize` or Dask for out-of-core processing.")

    # Detect map/filter/lambda usage that may produce lists (py3 map returns iterator, but sometimes wrapped in list)
    if "map(" in code and "list(" in code:
        issues.append("Wrapping `map()` in `list()` forces evaluation; consider using iterators downstream to keep memory low.")

    # Detect recursion heavy usage
    # Simple scan: if function calls itself by name inside body
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == func_name:
                    issues.append(f"Recursive function `{func_name}` detected — recursion depth may increase memory/stack usage; consider iterative approach or tail recursion optimization (not supported in CPython).")
                    break

    return issues

def analyze_js_static(code: str) -> List[str]:
    issues = []
    # heuristic checks
    if re.search(r"\bfor\s*\(.*;.*;.*\)\s*{", code):
        issues.append("Classic for-loop detected. For large data processing in JS consider streaming patterns or use typed arrays where appropriate.")
    if re.search(r"\bJSON\.parse\(", code):
        issues.append("JSON.parse loads full object into memory. For large payloads consider streaming JSON parser.")
    if re.search(r"\beval\(", code):
        issues.append("Use of eval() detected — security & performance risk.")
    if re.search(r"\bnew\s+Array\(", code):
        issues.append("Allocating large arrays may spike memory; if you only need streaming or iterating, consider other structures.")
    if re.search(r"\bforEach\(", code) and "map(" in code and "filter(" in code:
        issues.append("Chaining array methods may create intermediate arrays; consider using generators or reduce-style streaming when memory matters.")
    return issues

def analyze_java_static(code: str) -> List[str]:
    issues = []
    if re.search(r"new\s+ArrayList<", code):
        issues.append("ArrayList allocations detected — ensure you set initial capacity if you know size to avoid repeated resizing.")
    if re.search(r"\bString\.split\(", code):
        issues.append("String.split creates many String objects; for large strings consider streaming splitting or scanner approaches.")
    if re.search(r"\bfor\s*\(.*;.*;.*\)\s*{", code):
        issues.append("Classic for-loop detected — check algorithmic complexity for nested loops.")
    if "synchronized" in code:
        issues.append("Synchronized blocks can cause CPU waits/locking; ensure proper concurrency design.")
    return issues

def analyze_static(code: str, language: str) -> List[str]:
    if language == "python":
        return analyze_python_static(code)
    elif language == "javascript":
        return analyze_js_static(code)
    elif language == "java":
        return analyze_java_static(code)
    else:
        return ["Unsupported language for static analysis."]

# ---------------------------
# DYNAMIC EXECUTION & PROFILING
# ---------------------------

def write_temp_file(code: str, ext: str) -> str:
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path

def profile_subprocess_run(cmd: List[str], timeout: int = MAX_RUN_SECONDS) -> Dict[str, Any]:
    """
    Runs a subprocess command and samples its cpu/memory while it runs.
    Returns dict: success(bool), runtime_seconds, max_rss_kb, avg_cpu_percent, stdout, stderr, notes
    Requires psutil for sampling; if psutil not available, will measure time and return note.
    """
    result = {
        "success": False,
        "runtime_seconds": None,
        "max_rss_kb": None,
        "avg_cpu_percent": None,
        "stdout": "",
        "stderr": "",
        "notes": "",
    }

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        result["notes"] = f"Failed to spawn process: {e}"
        return result

    pid = proc.pid
    max_rss = 0
    cpu_samples = []
    ps_proc = None
    if psutil:
        try:
            ps_proc = psutil.Process(pid)
        except Exception:
            ps_proc = None

    try:
        # poll while running
        while True:
            if proc.poll() is not None:
                break
            now = time.time()
            if ps_proc:
                try:
                    mem = ps_proc.memory_info().rss // 1024  # KB
                    cpu = ps_proc.cpu_percent(interval=None)
                    max_rss = max(max_rss, mem)
                    cpu_samples.append(cpu)
                except Exception:
                    pass
            time.sleep(SAMPLING_INTERVAL)
            # check timeout
            if now - start > timeout:
                proc.kill()
                result["notes"] += f"Process killed after timeout {timeout}s. "
                break

        stdout, stderr = proc.communicate(timeout=1)
        end = time.time()
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["runtime_seconds"] = round(end - start, 4)
        result["max_rss_kb"] = int(max_rss) if max_rss else None
        result["avg_cpu_percent"] = round(sum(cpu_samples) / len(cpu_samples), 2) if cpu_samples else None
        result["success"] = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        result["notes"] += "Timeout during communicate; process killed. "
    except Exception as e:
        proc.kill()
        result["notes"] += f"Error while running process: {e}"
    return result

def run_python_dynamic(code: str) -> Dict[str, Any]:
    # Wrap user code to protect accidental long-running constructs? We will simply write to temp file and run with -u
    path = write_temp_file(code, ".py")
    cmd = [sys.executable, "-u", path]
    result = profile_subprocess_run(cmd)
    os.remove(path)
    return result

def run_node_dynamic(code: str) -> Dict[str, Any]:
    # requires node installed
    path = write_temp_file(code, ".js")
    cmd = ["node", path]
    result = profile_subprocess_run(cmd)
    os.remove(path)
    return result

def run_java_dynamic(code: str) -> Dict[str, Any]:
    # Write to .java, compile, run.
    # Expect a public class named Main or detect class name
    # Simple approach: wrap into class Main if no public class present
    class_name = None
    m = re.search(r"public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", code)
    if m:
        class_name = m.group(1)
        path = write_temp_file(code, ".java")
    else:
        # wrap into Main
        class_name = "Main"
        full = f"public class Main {{ public static void main(String[] args) throws Exception {{\n{code}\n}}}}"
        path = write_temp_file(full, ".java")
    workdir = tempfile.mkdtemp()
    src_path = os.path.join(workdir, f"{class_name}.java")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(code if m else full)
    try:
        # compile
        cp = subprocess.run(["javac", src_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if cp.returncode != 0:
            return {"success": False, "notes": "javac failed", "stdout": cp.stdout, "stderr": cp.stderr}
        # run
        cmd = ["java", "-cp", workdir, class_name]
        result = profile_subprocess_run(cmd)
        return result
    finally:
        # cleanup
        try:
            for fn in os.listdir(workdir):
                os.remove(os.path.join(workdir, fn))
            os.rmdir(workdir)
        except Exception:
            pass

# ---------------------------
# SUGGESTION GENERATOR
# ---------------------------
def generate_suggestions(static_issues: List[str], dynamic_metrics: Dict[str, Any], language: str) -> List[str]:
    suggestions = []

    # From static issues directly
    for issue in static_issues:
        suggestions.append(issue)

    # From dynamic metrics thresholds
    runtime = dynamic_metrics.get("runtime_seconds")
    max_rss = dynamic_metrics.get("max_rss_kb")
    cpu = dynamic_metrics.get("avg_cpu_percent")

    if runtime is not None:
        if runtime > 2.0:
            suggestions.append(f"Long execution time observed ({runtime}s). Profile hotspots (line-level) and consider algorithmic improvements (reduce complexity, avoid nested loops).")
        else:
            suggestions.append(f"Execution time looks short ({runtime}s).")

    if max_rss is not None:
        # heuristic thresholds (KB)
        if max_rss > 100_000:  # >100MB
            suggestions.append(f"High peak memory usage observed (~{max_rss//1024} MB). Consider streaming data, generators, or processing in chunks.")
        elif max_rss > 30_000:
            suggestions.append(f"Moderate peak memory usage (~{max_rss//1024} MB). Review large in-memory allocations and caches.")

    if cpu is not None:
        if cpu > 80:
            suggestions.append(f"High CPU usage sampled (~{cpu}%). Look for tight loops or CPU-bound operations and consider C-accelerated libraries or concurrency patterns appropriate for your language.")
        else:
            suggestions.append(f"Average CPU usage sampled (~{cpu}%).")

    # Language-specific hints
    if language == "python":
        suggestions.append("Python tips: prefer generator expressions, use itertools for streaming, use numpy/pandas vectorized ops for numerical data, and consider concurrency with multiprocessing for CPU-bound tasks.")
    elif language == "javascript":
        suggestions.append("JavaScript tips: avoid blocking synchronous operations in Node; use streams for file/HTTP processing and avoid building giant arrays in memory.")
    elif language == "java":
        suggestions.append("Java tips: pre-size collections, reuse buffers, prefer streaming API and avoid unnecessary object allocations in hot loops.")

    # Add sustainability note
    suggestions.append("Consider measuring energy consumption with external tools (e.g., Intel RAPL or hardware meters) for production-critical workloads — reducing CPU and memory usage directly reduces energy usage and cloud cost.")

    return suggestions

# ---------------------------
# STREAMLIT UI
# ---------------------------
def sidebar_history_ui():
    st.sidebar.header("Saved analyses")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, filename, language, runtime_seconds, max_rss_kb FROM analyses ORDER BY timestamp DESC LIMIT 10")
    rows = cur.fetchall()
    conn.close()
    history_labels = [f"{fname} ({lang}) — {ts[:19]} — {runtime}s / {rss}KB" for id_, ts, fname, lang, runtime, rss in rows]
    history_ids = [id_ for id_, ts, fname, lang, runtime, rss in rows]
    selected = st.sidebar.selectbox("Click a history item to view details", options=history_ids, format_func=lambda x: history_labels[history_ids.index(x)] if x in history_ids else str(x)) if history_ids else None
    return selected

def main():
    init_db()
    st.set_page_config(page_title="Green IT Code Analyzer", layout="wide")
    st.title("🌱 Green IT Code Analyzer")
    st.write("Paste or upload code (Python, JavaScript, Java). The tool runs static analysis + (optional) dynamic profiling, then gives optimization suggestions. **Run in a sandboxed environment for safety.**")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Analyze", "History"])

    if page == "Analyze":
        col1, col2 = st.columns([2,1])
        with col1:
            input_mode = st.radio("Input method", ["Paste code", "Upload file", "Upload folder (zip)"], horizontal=True)
            code = ""
            filename = ""
            language = "python"
            if input_mode == "Paste code":
                code = st.text_area("Paste your code here", height=320)
                filename = st.text_input("Filename (optional, e.g. script.py)", value="snippet")
            elif input_mode == "Upload file":
                uploaded = st.file_uploader("Upload code file", type=["py","js","java","txt"])
                if uploaded:
                    filename = uploaded.name
                    code = uploaded.getvalue().decode("utf-8")
                    st.code(code[:1000], language="python")
            else:
                uploaded_zip = st.file_uploader("Upload a zip file containing source files (folder)", type=["zip"])
                run_dynamic_for_folder = st.checkbox("Run dynamic profiling for Python files in folder (may execute code)", value=False)
                if uploaded_zip:
                    # process zip when user clicks Analyze
                    pass
            language = st.selectbox("Language", SUPPORTED_LANGS, index=0 if "python" in SUPPORTED_LANGS else 0)

            run_dynamic = st.checkbox("Run dynamic profiling (may execute code) — recommended for Python", value=True)
            run_button = st.button("Analyze")

        with col2:
            st.markdown("### Quick tips")
            st.markdown("- Dynamic profiling requires `psutil` for memory/CPU sampling.")
            st.markdown("- Java/Node must be installed for live profiling of Java/JS.")
            st.markdown("- If you only want static checks, uncheck dynamic profiling.")
            st.markdown("- The app stores reports in a local SQLite DB.")

        if run_button:
            # Validation depends on input mode
            if input_mode == "Upload folder (zip)":
                if 'uploaded_zip' not in locals() or not uploaded_zip:
                    st.error("Please upload a zip file to analyze.")
                    return
            else:
                if not code or code.strip() == "":
                    st.error("Please provide code to analyze.")
                    return

            st.info("Running static analysis...")
            static_issues = analyze_static(code, language)
            st.success(f"Static analysis found {len(static_issues)} issues.")
            for s in static_issues:
                st.write("- " + s)

            dynamic_metrics = {}
            notes = ""
            carbon_kg = None
            if run_dynamic:
                st.info("Running dynamic profiling (this will execute the code in a subprocess)...")
                if language == "python":
                    try:
                        if not psutil:
                            st.warning("psutil not installed — dynamic memory/CPU sampling disabled; only time will be measured.")
                        res = run_python_dynamic(code)
                        dynamic_metrics = {
                            "runtime_seconds": res.get("runtime_seconds"),
                            "max_rss_kb": res.get("max_rss_kb"),
                            "avg_cpu_percent": res.get("avg_cpu_percent"),
                        }
                        notes = res.get("notes","")
                        st.write("**Stdout:**")
                        st.code(res.get("stdout","")[:2000])
                        if res.get("stderr"):
                            st.write("**Stderr:**")
                            st.code(res.get("stderr","")[:2000])
                    except Exception as e:
                        st.error(f"Dynamic profiling failed: {e}")
                elif language == "javascript":
                    if shutil_which("node") is None:
                        st.warning("Node not found: skipping dynamic run. Install `node` if you want to profile JS.")
                        notes = "node not found"
                    else:
                        res = run_node_dynamic(code)
                        dynamic_metrics = {
                            "runtime_seconds": res.get("runtime_seconds"),
                            "max_rss_kb": res.get("max_rss_kb"),
                            "avg_cpu_percent": res.get("avg_cpu_percent"),
                        }
                        notes = res.get("notes","")
                        st.write("**Stdout:**")
                        st.code(res.get("stdout","")[:2000])
                        if res.get("stderr"):
                            st.write("**Stderr:**")
                            st.code(res.get("stderr","")[:2000])
                elif language == "java":
                    if shutil_which("javac") is None or shutil_which("java") is None:
                        st.warning("javac/java not found: skipping dynamic run. Install JDK if you want to profile Java.")
                        notes = "javac/java not found"
                    else:
                        res = run_java_dynamic(code)
                        dynamic_metrics = {
                            "runtime_seconds": res.get("runtime_seconds"),
                            "max_rss_kb": res.get("max_rss_kb"),
                            "avg_cpu_percent": res.get("avg_cpu_percent"),
                        }
                        notes = res.get("notes","")
                else:
                    st.warning("Dynamic profiling disabled for this language.")

            else:
                st.info("Dynamic profiling skipped by user.")

            # Estimate carbon emissions if possible
            if dynamic_metrics.get("runtime_seconds") and dynamic_metrics.get("avg_cpu_percent") is not None and dynamic_metrics.get("max_rss_kb") is not None:
                # Use measured values, fallback to defaults if missing
                cpu_power_w = 65
                cpu_util = (dynamic_metrics.get("avg_cpu_percent") or 80) / 100
                mem_gb = (dynamic_metrics.get("max_rss_kb") or 200000) / 1024 / 1024  # KB to GB
                if mem_gb == 0:
                    mem_gb = 0.2
                carbon_kg = estimate_carbon_emissions(
                    runtime_seconds=dynamic_metrics.get("runtime_seconds"),
                    cpu_power_w=cpu_power_w,
                    cpu_util=cpu_util,
                    mem_gb=mem_gb,
                    mem_power_per_gb=3,
                    carbon_intensity=0.475
                )

            # Generate suggestions
            st.info("Generating suggestions...")
            suggestions = generate_suggestions(static_issues, dynamic_metrics, language)
            st.header("Optimization Suggestions")
            for s in suggestions:
                st.write("- " + s)

            # Show estimated carbon emissions
            if carbon_kg is not None:
                st.subheader("Estimated Carbon Emissions")
                st.write(f"{carbon_kg*1000:.2f} grams CO₂e (global average)")

            # Save to DB
            rec = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "filename": filename or f"snippet_{rec_id_short()}",
                "language": language,
                "static_issues": static_issues,
                "runtime_seconds": dynamic_metrics.get("runtime_seconds"),
                "max_rss_kb": dynamic_metrics.get("max_rss_kb"),
                "avg_cpu_percent": dynamic_metrics.get("avg_cpu_percent"),
                "success": True,
                "notes": notes,
            }
            save_analysis(rec)
            st.success("Analysis saved to local SQLite history.")
        # Handle folder zip analysis
        if input_mode == "Upload folder (zip)" and 'uploaded_zip' in locals() and uploaded_zip and run_button:
            # Extract zip to tempdir
            tmpdir = tempfile.mkdtemp()
            try:
                zpath = os.path.join(tmpdir, "upload.zip")
                with open(zpath, "wb") as f:
                    f.write(uploaded_zip.getvalue())
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(tmpdir)

                # Walk extracted files and analyze supported files
                supported_ext = {".py": "python", ".js": "javascript", ".java": "java"}
                results = []
                max_files = 500
                count = 0
                for root, dirs, files in os.walk(tmpdir):
                    for fname in files:
                        if count >= max_files:
                            break
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in supported_ext:
                            count += 1
                            fpath = os.path.join(root, fname)
                            rel = os.path.relpath(fpath, tmpdir)
                            try:
                                with open(fpath, "r", encoding="utf-8", errors="ignore") as rf:
                                    code_txt = rf.read()
                            except Exception:
                                code_txt = ""

                            lang = supported_ext[ext]
                            static_issues = analyze_static(code_txt, lang)
                            dynamic_metrics = {}
                            notes = ""
                            carbon_kg = None
                            if run_dynamic_for_folder and lang == "python":
                                try:
                                    res = run_python_dynamic(code_txt)
                                    dynamic_metrics = {
                                        "runtime_seconds": res.get("runtime_seconds"),
                                        "max_rss_kb": res.get("max_rss_kb"),
                                        "avg_cpu_percent": res.get("avg_cpu_percent"),
                                    }
                                    notes = res.get("notes", "")
                                except Exception as e:
                                    notes = str(e)

                            # Estimate carbon if metrics available
                            if dynamic_metrics.get("runtime_seconds"):
                                cpu_power_w = 65
                                cpu_util = (dynamic_metrics.get("avg_cpu_percent") or 80) / 100
                                mem_gb = (dynamic_metrics.get("max_rss_kb") or 200000) / 1024 / 1024
                                if mem_gb == 0:
                                    mem_gb = 0.2
                                carbon_kg = estimate_carbon_emissions(
                                    runtime_seconds=dynamic_metrics.get("runtime_seconds"),
                                    cpu_power_w=cpu_power_w,
                                    cpu_util=cpu_util,
                                    mem_gb=mem_gb,
                                    mem_power_per_gb=3,
                                    carbon_intensity=0.475,
                                )

                            # Save per-file analysis
                            rec = {
                                "id": str(uuid.uuid4()),
                                "timestamp": datetime.utcnow().isoformat(),
                                "filename": rel,
                                "language": lang,
                                "static_issues": static_issues,
                                "runtime_seconds": dynamic_metrics.get("runtime_seconds"),
                                "max_rss_kb": dynamic_metrics.get("max_rss_kb"),
                                "avg_cpu_percent": dynamic_metrics.get("avg_cpu_percent"),
                                "success": True,
                                "notes": notes,
                            }
                            save_analysis(rec)

                            # Append result for display
                            results.append({
                                "filename": rel,
                                "language": lang,
                                "static_issues": static_issues,
                                "dynamic": dynamic_metrics,
                                "carbon_kg": carbon_kg,
                                "notes": notes,
                            })

                # Display results neatly
                st.header(f"Folder analysis results ({len(results)} files analyzed)")
                for r in results:
                    with st.expander(r["filename"]):
                        st.write(f"**Language:** {r['language']}")
                        st.subheader("Static issues")
                        if r["static_issues"]:
                            for it in r["static_issues"]:
                                st.write("- " + it)
                        else:
                            st.write("No static issues detected.")
                        if r["dynamic"]:
                            st.subheader("Dynamic metrics")
                            st.write(r["dynamic"]) 
                        if r["carbon_kg"] is not None:
                            st.subheader("Estimated Carbon Emissions")
                            st.write(f"{r['carbon_kg']*1000:.3f} grams CO₂e")
                        if r["notes"]:
                            st.write(f"Notes: {r['notes']}")

            finally:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

    elif page == "History":
        selected_history_id = sidebar_history_ui()
        if selected_history_id:
            # Show details for selected history item
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT timestamp, filename, language, static_issues, runtime_seconds, max_rss_kb, avg_cpu_percent, success, notes FROM analyses WHERE id=?", (selected_history_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                ts, fn, lang, static_issues, runtime, rss, cpu, success, notes = row
                st.header(f"Analysis Details: {fn} ({lang})")
                st.write(f"**Timestamp:** {ts}")
                st.write(f"**Runtime:** {runtime}s")
                st.write(f"**Peak Memory:** {rss} KB")
                st.write(f"**Avg CPU:** {cpu}%")
                st.write(f"**Success:** {'Yes' if success else 'No'}")
                st.write(f"**Notes:** {notes}")
                st.subheader("Static Issues")
                for issue in static_issues.split('\n'):
                    st.write(f"- {issue}")

# small helpers
def rec_id_short():
    return uuid.uuid4().hex[:8]

def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)

if __name__ == "__main__":
    main()

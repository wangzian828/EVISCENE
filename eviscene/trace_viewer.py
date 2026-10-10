"""Dependency-free standalone HTML renderer for unified-ReAct traces."""
from __future__ import annotations

import base64
import html
import json
import os
from typing import Any, Dict

from eviscene.redaction import sanitize_for_persistence


def save_trace_html(trace_data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(_render_trace_html(sanitize_for_persistence(trace_data)))


def render_trace_file(json_path: str, output_path: str) -> None:
    with open(json_path, encoding="utf-8") as handle:
        save_trace_html(json.load(handle), output_path)


def _render_trace_html(trace_data: Dict[str, Any]) -> str:
    state = trace_data.get("state") if isinstance(trace_data.get("state"), dict) else {}
    image_path = str(trace_data.get("image_path") or state.get("image_path") or "")
    judgment = trace_data.get("judgment") if isinstance(trace_data.get("judgment"), dict) else {}
    if not judgment and isinstance(state.get("judgment"), dict):
        judgment = state["judgment"]
    steps = state.get("all_steps", []) if isinstance(state.get("all_steps"), list) else []
    investigation = state.get("investigation_state", {}) if isinstance(state.get("investigation_state"), dict) else {}
    timings = state.get("stage_timings", {}) if isinstance(state.get("stage_timings"), dict) else {}
    token_usage = trace_data.get("token_usage") or state.get("token_usage") or {}
    embedded = _embed_local_image(image_path)
    stage_flow = (
        '<div class="stage">Unified ReAct</div><span class="arrow">&#8594;</span>'
        '<div class="stage">unified-react Judgment</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Verification Trace</title>
<style>
:root{{--ink:#17202a;--muted:#5e6b75;--line:#d8dee4;--paper:#fff;--bg:#f4f6f8;--blue:#1769aa;--green:#26734d;--red:#b42318;--amber:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;font-size:14px;letter-spacing:0}}
.wrap{{max-width:1280px;margin:auto;padding:20px}}h1{{font-size:26px;margin:0 0 14px}}h2{{font-size:18px;margin:0 0 12px}}h3{{font-size:14px;margin:14px 0 7px}}
.band{{background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:16px;margin-bottom:14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.kv{{min-width:0}}.kv b{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:3px}}.pipeline{{display:flex;gap:8px;align-items:center;overflow:auto;padding:4px 0}}
.stage{{border-left:4px solid var(--blue);background:#f7fafc;padding:8px 10px;min-width:135px}}.arrow{{color:var(--muted)}}.two{{display:grid;grid-template-columns:minmax(260px,360px) 1fr;gap:14px}}
.image img{{display:block;max-width:100%;max-height:440px;border:1px solid var(--line)}}.revision,.audit,.step{{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}}
.revision:first-child,.audit:first-child,.step:first-child{{border-top:0;padding-top:0;margin-top:0}}.badge{{display:inline-block;background:#e8f1f8;color:#174f78;padding:3px 7px;border-radius:4px;margin:0 5px 4px 0;font-size:11px}}
.ok{{background:#e7f6ec;color:var(--green)}}.bad{{background:#feeceb;color:var(--red)}}.warn{{background:#fff3cd;color:var(--amber)}}code{{font-family:Consolas,monospace}}pre{{white-space:pre-wrap;word-break:break-word;overflow:auto;background:#f6f8fa;border:1px solid #e7eaee;padding:10px;margin:7px 0 0;font-size:12px}}
.interaction-chain{{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;align-items:stretch;margin:8px 0 12px;padding:9px;background:#f6f8fa;border:1px solid #e7eaee}}.interaction-node{{min-width:0;padding:6px 8px;border-left:3px solid var(--line);background:var(--paper)}}.interaction-node.current{{border-left-color:var(--blue)}}.interaction-key{{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}}.interaction-node code{{display:block;white-space:normal;overflow-wrap:anywhere;word-break:break-word}}.chain-arrow{{align-self:center;color:var(--muted);font-size:18px;line-height:1}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}th{{background:#f6f8fa}}ul{{margin:6px 0;padding-left:20px}}
@media(max-width:760px){{.wrap{{padding:10px}}.two{{grid-template-columns:1fr}}.pipeline{{align-items:stretch}}.stage{{min-width:125px}}.interaction-chain{{grid-template-columns:minmax(0,1fr)}}.chain-arrow{{justify-self:center;transform:rotate(90deg)}}}}
</style>
</head>
<body><main class="wrap">
<section class="band"><h1>Image Verification Trace</h1>{_render_meta(trace_data, state, judgment, image_path, token_usage)}</section>
<section class="band"><h2>Stage Flow</h2><div class="pipeline">{stage_flow}</div></section>
<div class="two"><div>{_render_image(embedded)}</div><div>
{_render_judgment(judgment, trace_data)}{_render_timings(timings)}
</div></div>
{_render_runtime_state(investigation)}
<section class="band"><h2>Agent Trajectory</h2>{''.join(_render_step(step) for step in steps if isinstance(step, dict)) or '<p>No recorded steps.</p>'}</section>
</main></body></html>"""


def _render_meta(trace: Dict[str, Any], state: Dict[str, Any], judgment: Dict[str, Any], image_path: str, tokens: Dict[str, Any]) -> str:
    values = [
        ("Image", os.path.basename(image_path) or "unknown"),
        ("Verdict", judgment.get("verdict", trace.get("verdict", "unverifiable"))),
        ("Confidence", _number(judgment.get("confidence", trace.get("confidence", 0.0)), 2)),
        ("Termination", trace.get("termination", state.get("termination", ""))),
        ("Tool Calls", trace.get("total_tool_calls", state.get("total_tool_calls", 0))),
        ("LLM Calls", trace.get("llm_api_calls", state.get("llm_api_calls", 0))),
        ("Tokens", f"{tokens.get('prompt', 0)} in / {tokens.get('completion', 0)} out"),
        ("Time", f"{_number(trace.get('time_taken', 0.0), 1)}s"),
    ]
    return '<div class="grid">' + ''.join(f'<div class="kv"><b>{_e(key)}</b>{_e(value)}</div>' for key, value in values) + '</div>'


def _render_image(data_url: str) -> str:
    if not data_url:
        return '<section class="band"><h2>Input Image</h2><p>Image is not available locally.</p></section>'
    return f'<section class="band image"><h2>Input Image</h2><img src="{data_url}" alt="Input image"></section>'


def _render_judgment(judgment: Dict[str, Any], trace: Dict[str, Any]) -> str:
    assessment = judgment.get("overall_assessment", trace.get("overall_assessment", ""))
    basis = trace.get("verdict_basis")
    report = judgment.get("fact_check_report")
    citations = judgment.get("evidence_citations")
    report_html = ""
    if isinstance(report, dict):
        report_html = (
            '<section class="subband"><h3>Fact-check Report</h3>'
            + f'<h4>{_e(report.get("headline", ""))}</h4>'
            + f'<p><b>Claim under review:</b> {_e(report.get("claim_under_review", ""))}</p>'
            + f'<p><b>Conclusion:</b> {_e(report.get("verdict_summary", ""))}</p>'
            + _list("Key findings", report.get("key_findings"))
            + f'<p><b>Evidence summary:</b> {_e(report.get("evidence_summary", ""))}</p>'
            + _list("Remaining uncertainties", report.get("remaining_uncertainties"))
            + _labeled_json("Evidence citations", citations)
            + '</section>'
        )
    return (
        '<section class="band"><h2>Final Judgment</h2>'
        + f'<p>{_e(assessment)}</p>'
        + report_html
        + _labeled_json("Verdict Basis", basis)
        + '</section>'
    )


def _render_step(step: Dict[str, Any]) -> str:
    metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
    action = str(step.get("action_type", ""))
    badge_class = "bad" if action in {"format_error", "output_rejected"} else ("ok" if action in {"tool_call", "output"} else "")
    labels = [
        f'<span class="badge">{_e(step.get("stage", "unknown"))}</span>',
        f'<span class="badge">Round {_e(step.get("round", "?"))}</span>',
        f'<span class="badge {badge_class}">{_e(action)}</span>',
    ]
    if metadata.get("verification_iteration") is not None:
        labels.append(f'<span class="badge">Verification iteration {_e(metadata["verification_iteration"])}</span>')
    if metadata.get("native_interactions"):
        labels.append('<span class="badge">Gemini Interactions</span>')
    if metadata.get("forced_output"):
        labels.append('<span class="badge warn">Forced output</span>')
    if metadata.get("rejection_reason"):
        labels.append(f'<span class="badge bad">{_e(metadata["rejection_reason"])}</span>')
    tool_name = step.get("tool_name", "")
    body = f'<h3>{_e(tool_name or "Structured output")}</h3>'
    body += _render_interaction_chain(metadata)
    body += _labeled_pre("Thought", step.get("thought"))
    body += _labeled_json("Arguments", step.get("tool_args"))
    body += _labeled_pre("Result", step.get("tool_result"))
    body += _labeled_json("Output", step.get("output"))
    return '<article class="step"><div>' + ''.join(labels) + '</div>' + body + '</article>'


def _render_runtime_state(investigation: Dict[str, Any]) -> str:
    if not investigation:
        return (
            '<section class="band"><h2>Raw ReAct Runtime</h2>'
            '<p>No investigation state.</p></section>'
        )
    fields = {
        key: investigation.get(key)
        for key in (
            "schema_version",
            "case_id",
            "image_sha256",
            "objective",
            "action_count",
            "stop_reason",
            "finish_rationale",
        )
        if investigation.get(key) not in (None, "", [], {})
    }
    return (
        '<section class="band"><h2>Raw ReAct Runtime</h2>'
        + _labeled_json("Mechanical state", fields)
        + "</section>"
    )


def _render_interaction_chain(metadata: Dict[str, Any]) -> str:
    if not metadata.get("native_interactions"):
        return ""
    parent_recorded = "previous_interaction_id" in metadata
    parent = metadata.get("previous_interaction_id")
    interaction_id = metadata.get("interaction_id")
    if not parent_recorded:
        parent_text = "not recorded"
    else:
        parent_text = "null (root)" if parent in (None, "") else str(parent)
    child_text = str(interaction_id) if interaction_id not in (None, "") else "missing"
    return (
        '<div class="interaction-chain" aria-label="Interaction request chain">'
        '<div class="interaction-node parent"><span class="interaction-key">'
        'previous_interaction_id</span><code>'
        + _e(parent_text)
        + '</code></div><span class="chain-arrow" aria-hidden="true">&#8594;</span>'
        '<div class="interaction-node current"><span class="interaction-key">'
        'interaction_id</span><code>'
        + _e(child_text)
        + "</code></div></div>"
    )


def _render_timings(timings: Dict[str, Any]) -> str:
    if not timings:
        return ""
    return '<section class="band"><h2>Stage Timings</h2>' + _list("", [f"{key}: {value}s" for key, value in timings.items()]) + '</section>'


def _list(title: str, items: Any) -> str:
    if not isinstance(items, list) or not items:
        return ""
    heading = f'<h3>{_e(title)}</h3>' if title else ""
    return heading + '<ul>' + ''.join(f'<li>{_e(item)}</li>' for item in items) + '</ul>'


def _labeled_json(label: str, value: Any) -> str:
    if value in (None, {}, []):
        return ""
    return f'<b>{_e(label)}</b>' + _pre(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _labeled_pre(label: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<b>{_e(label)}</b>' + _pre(value)


def _pre(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<pre>{_e(value)}</pre>'


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _embed_local_image(image_path: str) -> str:
    if not image_path or not os.path.isfile(image_path):
        return ""
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(os.path.splitext(image_path)[1].lower(), "image/jpeg")
    return f"data:{mime};base64,{encoded}"

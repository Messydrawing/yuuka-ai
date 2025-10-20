import os
import sys
import importlib.util
from pathlib import Path

import pytest


def import_app_module():
    os.environ.setdefault("YUKA_SKIP_MODEL_LOAD", "1")
    module_name = "app_yuuka_ui"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "app_yuuka_ui.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_structured_summary_extracts_fields():
    app = import_app_module()
    summary_text = (
        "【对话概览】\n"
        "- 老师布置了新的盘点任务\n"
        "- 优香确认了时间安排\n"
        "【当前优香对老师的态度/情绪】温柔且细致\n"
        "【关系进展要点】\n"
        "- 双方约定下周复盘进度\n"
    )
    sections = app.parse_structured_summary(summary_text)
    assert sections["overview"] == ["老师布置了新的盘点任务", "优香确认了时间安排"]
    assert sections["attitude"] == "温柔且细致"
    assert sections["progress"] == ["双方约定下周复盘进度"]


def test_build_memory_context_renders_attitude_line():
    app = import_app_module()
    mem = app.new_memory("test")
    mem["turns"] = [
        {"u": "老师说明即将盘点", "a": "好的老师，我来准备"},
        {"u": "老师询问心情", "a": "稍微有点紧张"},
    ]
    structured_summary = (
        "【对话概览】\n"
        "- 老师布置财务核对\n"
        "【当前优香对老师的态度/情绪】保持耐心与支持\n"
        "【关系进展要点】\n"
        "- 约定明日提交初稿\n"
    )
    mem["block_summaries"].append({
        "start": 0,
        "end": 1,
        "summary": structured_summary,
    })
    context_text = app.build_memory_context(mem)
    assert "当前优香对老师的态度/情绪" in context_text
    assert "保持耐心与支持" in context_text
    assert "关系进展要点" in context_text
    assert "约定明日提交初稿" in context_text


def test_keyword_memory_vector_search(tmp_path):
    app = import_app_module()
    if getattr(app, "np", None) is None:
        pytest.skip("numpy is required for vector search")

    kb_file = tmp_path / "memory.jsonl"
    kb_file.write_text("", encoding="utf-8")
    km = app.KeywordMemory(kb_file)

    mem = app.new_memory("session")
    mem["turns"] = [
        {"u": "老师提醒预算审核进度，需要今晚提交调整方案", "a": "优香确认会在晚上十点前整理预算明细"},
        {"u": "老师追问报销凭证是否齐全", "a": "优香表示欠缺一张交通票据"},
    ]
    mem["block_summaries"].append({
        "start": 0,
        "end": 1,
        "summary": "老师与优香对预算审核进行梳理，确认今晚提交调整方案并补齐报销凭证",
    })
    km.queue_memory_fragments(mem)
    km.flush_pending(force=True)

    hits = km.search_similar_fragments("预算审核进度怎么样了？", top_k=2, min_score=0.0)
    assert hits
    assert any("预算" in (hit.get("label") or hit.get("content") or "") for hit in hits)

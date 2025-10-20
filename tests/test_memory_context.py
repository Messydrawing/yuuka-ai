import os
import sys
import importlib.util
from pathlib import Path


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

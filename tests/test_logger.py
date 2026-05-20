import importlib
import logging
import sys
from pathlib import Path


def test_logger_writes_to_logs_directory(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    src_dir = project_root / "src"
    app_dir = src_dir / "app"
    app_dir.mkdir(parents=True)

    # Make app.logger compute the project root from this temporary src/app path.
    real_logger_file = Path(__file__).resolve().parents[1] / "src" / "app" / "logger.py"
    logger_source = real_logger_file.read_text(encoding="utf-8")
    temp_logger_file = app_dir / "logger.py"
    temp_logger_file.write_text(logger_source, encoding="utf-8")
    (app_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.syspath_prepend(str(src_dir))
    sys.modules.pop("app.logger", None)
    old_app_module = sys.modules.pop("app", None)

    try:
        module = importlib.import_module("app.logger")
    finally:
        if old_app_module is not None:
            sys.modules["app"] = old_app_module
    test_logger = logging.getLogger("app.test.file")
    test_logger.info("file logging smoke test")

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = project_root / "logs" / "webai-to-api.log"
    assert log_file.exists()
    assert "file logging smoke test" in log_file.read_text(encoding="utf-8")

    # Keep module referenced so import side effects are explicit for linters/readability.
    assert module.logger.name == "app"

from app import config as app_config
from app.utils.browser import CrossPlatformCookieExtractor


def test_cdp_port_defaults_to_9222_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WEBAI_CDP_PORT", raising=False)
    monkeypatch.setattr(app_config, "CONFIG", app_config.load_config("missing.conf"))
    import app.utils.browser as browser_module
    monkeypatch.setattr(browser_module, "CONFIG", app_config.CONFIG)
    extractor = CrossPlatformCookieExtractor()

    assert extractor._get_cdp_port() == 9222


def test_cdp_port_can_be_read_from_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WEBAI_CDP_PORT", raising=False)
    (tmp_path / "config.conf").write_text("[Browser]\ncdp_port = 9223\n", encoding="utf-8")
    extractor = CrossPlatformCookieExtractor()

    assert extractor._get_cdp_port() == 9223

"""aikit 单元测试（不发起真实网络请求）。

运行：python -m pytest tests/ -q   或   python tests/test_core.py
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aikit import (  # noqa: E402
    AI, ConfigStore, GenParams, Message, StoredChannel, mask_key,
)
from aikit.client import _build_payload, _extract_json, _parse_choice  # noqa: E402
from aikit.models import ModelConfig  # noqa: E402


def test_mask_key():
    assert mask_key("") == ""
    assert mask_key("short") == "*****"
    out = mask_key("sk-1234567890abcdef")
    assert out.startswith("sk-1") and out.endswith("cdef") and "*" in out
    assert "1234567890ab" not in out


def test_message_normalize():
    assert Message.normalize("hi").role == "user"
    assert Message.normalize({"role": "system", "content": "x"}).content == "x"
    assert Message.normalize(Message("assistant", "y")).role == "assistant"


def test_extract_json():
    assert _extract_json('{"a":1}') == {"a": 1}
    assert _extract_json('```json\n{"b":[1,2]}\n```') == {"b": [1, 2]}
    assert _extract_json('好的，结果是 {"x": "中文"} 请查收') == {"x": "中文"}
    assert _extract_json("[1,2,3]") == [1, 2, 3]
    assert _extract_json("不是json") is None


def test_build_payload_basic_and_json():
    cfg = ModelConfig(api_key="k", base_url="https://x/v1", model="m")
    p = _build_payload(cfg, [{"role": "user", "content": "hi"}],
                       GenParams(temperature=0.5, max_tokens=10), False, None)
    assert p["model"] == "m"
    assert p["temperature"] == 0.5 and p["max_tokens"] == 10
    assert "response_format" not in p
    pj = _build_payload(cfg, [], None, True, None)
    assert pj["response_format"] == {"type": "json_object"}
    # None 参数不应出现
    assert "top_p" not in p


def test_parse_choice():
    data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                    "total_tokens": 2}}
    text, finish, usage = _parse_choice(data)
    assert text == "ok" and finish == "stop" and usage.total_tokens == 2


def test_config_env_source():
    os.environ["AIKIT_API_KEY"] = "sk-env-key-123456"
    os.environ["AIKIT_PROVIDER"] = "deepseek"
    os.environ.pop("AIKIT_BASE_URL", None)
    os.environ.pop("AIKIT_MODEL", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            store = ConfigStore(path=os.path.join(d, "c.json"))
            cfg = store.resolve()
            assert "deepseek.com" in cfg.base_url
            assert cfg.api_key == "sk-env-key-123456"
            assert cfg.model  # 预设默认模型
    finally:
        for k in ("AIKIT_API_KEY", "AIKIT_PROVIDER", "AIKIT_BASE_URL", "AIKIT_MODEL"):
            os.environ.pop(k, None)


def test_config_file_and_priority_and_mask():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        store = ConfigStore(path=path)
        store.upsert_channel("ds", StoredChannel(
            api_key="sk-file-secret-9999", provider="deepseek"),
            make_default=True)
        store.upsert_channel("xm", StoredChannel(
            api_key="sk-xm-0000", provider="xiaomi", model="mimo-v2-pro"))
        store.save()
        # 重新加载
        store2 = ConfigStore(path=path, use_env=False)
        assert store2.default == "ds"
        cfg = store2.resolve()
        assert cfg.provider == "deepseek"
        view = store2.public_view()
        # 安全视图不含明文
        assert "sk-file-secret-9999" not in json.dumps(view, ensure_ascii=False)
        assert view["channels"]["ds"]["has_key"] is True
        assert "*" in view["channels"]["ds"]["key_hint"]
        # 预设表里有 deepseek/xiaomi/custom
        ids = [p["id"] for p in view["providers"]]
        assert {"deepseek", "xiaomi", "custom"} <= set(ids)


def test_config_missing_raises():
    from aikit.errors import ConfigError
    with tempfile.TemporaryDirectory() as d:
        for k in ("AIKIT_API_KEY", "AIKIT_PROVIDER"):
            os.environ.pop(k, None)
        store = ConfigStore(path=os.path.join(d, "empty.json"), use_env=False)
        try:
            store.resolve()
            assert False, "应抛 ConfigError"
        except ConfigError:
            pass


class _FakeOne:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    async def raw_completion(self, payload):
        self.calls += 1
        item = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_chat_json_extract_with_fence():
    async def main():
        with tempfile.TemporaryDirectory() as d:
            store = ConfigStore(path=os.path.join(d, "c.json"), use_env=False)
            store.upsert_channel("a", StoredChannel(
                api_key="k", provider="custom",
                base_url="https://x/v1", model="m"), make_default=True)
            ai = AI(config=store, auto_failover=False)
            fake_resp = {"choices": [{"message": {
                "content": '```json\n{"answer":"C"}\n```'}, "finish_reason": "stop"}]}
            # monkeypatch _OneClient.raw_completion
            import aikit.client as C
            orig = C._OneClient.raw_completion
            async def fake_raw(self, payload):
                return fake_resp
            C._OneClient.raw_completion = fake_raw
            try:
                data = await ai.chat_json("选C", schema_hint='{"answer":str}')
                assert data == {"answer": "C"}
            finally:
                C._OneClient.raw_completion = orig
    asyncio.run(main())


if __name__ == "__main__":
    import inspect
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and inspect.isfunction(v)]
    passed = 0
    for fn in funcs:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception as exc:  # noqa
            print("FAIL", fn.__name__, repr(exc))
    print(f"\n{passed}/{len(funcs)} passed")

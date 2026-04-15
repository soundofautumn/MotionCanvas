import json

import pytest


def test_extract_json_object_direct():
    from apps.gradio.motioncanvas import _extract_json_object

    obj = _extract_json_object('{"a": 1, "b": [2, 3]}')
    assert obj["a"] == 1
    assert obj["b"] == [2, 3]


def test_extract_json_object_code_fence():
    from apps.gradio.motioncanvas import _extract_json_object

    text = """Here is json:
```json
{"updates": {"prompt": "hi"}}
```
"""
    obj = _extract_json_object(text)
    assert obj["updates"]["prompt"] == "hi"


def test_openai_sdk_base_url_normalization_applied_in_helper(monkeypatch):
    """No network: replace OpenAI SDK with a fake and verify base_url gets '/v1'."""
    from apps.gradio import motioncanvas as m

    created = {}

    class _FakeResp:
        def __init__(self, content):
            self._content = content

        def model_dump(self):
            return {"choices": [{"message": {"content": self._content}}]}

    class _FakeChatCompletions:
        def create(self, **kwargs):
            # return JSON content
            return _FakeResp('{"assistant_message":"ok","updates":{}}')

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeChatCompletions()

    class _FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            created["base_url"] = base_url
            created["api_key"] = api_key
            created["timeout"] = timeout
            self.chat = _FakeChat()

    # Patch import inside helper
    monkeypatch.setitem(__import__("sys").modules, "openai", type("X", (), {"OpenAI": _FakeOpenAI}))

    resp = m._openai_chat_complete(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        timeout=12,
        force_json=True,
    )
    assert created["base_url"].endswith("/v1")
    assert created["base_url"] == "https://api.deepseek.com/v1"
    assert resp["choices"][0]["message"]["content"]


def test_llm_apply_instruction_applies_updates(monkeypatch):
    from apps.gradio import motioncanvas as m

    # Patch the completion call to avoid network and return a well-formed JSON response
    def _fake_complete(**kwargs):
        content = json.dumps(
            {
                "assistant_message": "applied",
                "updates": {
                    "prompt": "new prompt",
                    "num_frames": 9,
                    "bbox_json": {"objects": [{"frames": {"0": [0.1, 0.1, 0.2, 0.2]}}]},
                },
            },
            ensure_ascii=False,
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(m, "_openai_chat_complete", lambda **kw: _fake_complete(**kw))

    out = m.llm_apply_instruction(
        user_message="make it",
        chat_history=[],
        llm_base_url="https://api.deepseek.com",
        llm_api_key="",
        llm_model="deepseek-chat",
        llm_timeout=30,
        bbox_json_text="",
        camera_json_text="",
        point_json_text="",
        prompt="old",
        negative_prompt="neg",
        height=480,
        width=832,
        num_frames=49,
        fps=15,
        num_inference_steps=50,
        cfg_scale=5.0,
        sigma_shift=5.0,
        seed=42,
        motion_frame_idx=0,
        bbox_kf_state={},
        point_kf_state={},
        camera_kf_state={},
    )

    (
        history,
        new_bbox_json,
        new_point_json,
        new_camera_json,
        new_bbox_state,
        new_point_state,
        new_camera_state,
        new_prompt,
        new_negative_prompt,
        new_height,
        new_width,
        new_num_frames,
        new_fps,
        new_steps,
        new_cfg,
        new_sigma,
        new_seed,
        frame_update,
        llm_status,
        cleared_msg,
    ) = out

    assert history and history[-1][1] == "applied"
    assert new_prompt == "new prompt"
    assert "objects" in json.loads(new_bbox_json)
    assert new_bbox_state.get("0") == [0.1, 0.1, 0.2, 0.2]
    # num_frames snaps to step=4 starting at 5 → 9 snaps to 9? (min=5 step=4 => 5,9,13...)
    assert int(new_num_frames) == 9
    assert "已应用" in llm_status
    assert cleared_msg == ""
    # frame slider update should cap maximum to num_frames-1
    assert frame_update["maximum"] == 8

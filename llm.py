import yaml
from openai import OpenAI
from pathlib import Path
import json
from collections import defaultdict

class LLM:
    def __init__(self, api_key: str, model: str = "gpt-5-mini", prompts_file: str = "prompts.yaml", track_usage: bool = False, usage_file: str = "token_usage.json"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompts = self._load_prompts(prompts_file)

        # Token tracking config
        self.track_usage = track_usage
        self.usage_file = usage_file
        self.usage_log = self._load_usage(usage_file)

    def _load_prompts(self, path: str):
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)

    def _load_usage(self, path: str):
        """Load existing usage file if available, else return defaultdict."""
        if Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
        return defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def build_messages(self, prompt_name: str, **kwargs):
        """Fill placeholders in YAML prompt and return messages list"""
        prompt_spec = self.prompts[prompt_name]

        system_msg = prompt_spec.get("system", "").format(**kwargs)
        user_msg = prompt_spec.get("user", "").format(**kwargs)

        messages = []
        if system_msg.strip():
            messages.append({"role": "system", "content": system_msg})
        if user_msg.strip():
            messages.append({"role": "user", "content": user_msg})

        return messages

    def call(self, prompt_name: str, **kwargs):
        messages = self.build_messages(prompt_name, **kwargs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        # Track usage if enabled
        if self.track_usage and response.usage:
            usage = response.usage
            log = self.usage_log.setdefault(prompt_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            log["prompt_tokens"] += usage.prompt_tokens
            log["completion_tokens"] += usage.completion_tokens
            log["total_tokens"] += usage.total_tokens
            self.save_usage()  # persist immediately after each call

        return response.choices[0].message.content

    def get_usage(self, prompt_name=None):
        """Return usage stats for all prompts or a single prompt."""
        if prompt_name:
            return dict(self.usage_log.get(prompt_name, {}))
        return {k: dict(v) for k, v in self.usage_log.items()}

    def save_usage(self):
        """Save usage stats to JSON file."""
        with open(self.usage_file, "w") as f:
            json.dump(self.usage_log, f, indent=2)

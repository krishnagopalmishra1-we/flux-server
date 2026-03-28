import json
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def call_generate(api_key: str, model_name: str) -> dict:
    payload = {
        "prompt": "portrait photo of a person in natural light",
        "model_name": model_name,
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.0 if model_name == "realvisxl-v5" else 6.5 if model_name == "juggernaut-xl" else 5.0,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            body = response.read().decode("utf-8")
            return {"http_status": response.status, "body": json.loads(body)}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {"http_status": error.code, "body": body}
    except Exception as error:
        return {"http_status": 0, "body": str(error)}


def main() -> None:
    env = load_env("/opt/flux-server/.env")
    api_key = env["API_KEYS"].split(",", 1)[0]
    for model_name in ["realvisxl-v5", "juggernaut-xl", "sd3-medium"]:
        result = call_generate(api_key, model_name)
        print(json.dumps({"model": model_name, **result}, ensure_ascii=True))


if __name__ == "__main__":
    main()
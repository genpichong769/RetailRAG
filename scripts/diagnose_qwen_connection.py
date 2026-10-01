"""Diagnose DashScope connectivity without loading any RetailRAG models or data."""

import json
import os
import socket
import ssl
import sys
import traceback
from pathlib import Path
from urllib.request import getproxies
from urllib.parse import urlparse

import httpx
import openai
import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config


def safe_proxy(name):
    value = os.getenv(name)
    if not value:
        return {"present": False, "endpoint": None}
    parsed = urlparse(value)
    endpoint = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}" if parsed.hostname else "configured"
    return {"present": True, "endpoint": endpoint}


def safe_system_proxies():
    output = {}
    for name, value in getproxies().items():
        parsed = urlparse(value)
        output[name] = {
            "scheme": parsed.scheme or None,
            "host": parsed.hostname,
            "port": parsed.port,
        }
    return output


def main():
    config = Config()
    parsed = urlparse(config.DASHSCOPE_BASE_URL)
    host = parsed.hostname or "dashscope.aliyuncs.com"
    result = {
        "status": "FAIL",
        "api_key": "PRESENT" if config.DASHSCOPE_API_KEY.strip() else "MISSING",
        "credential_value_logged": False,
        "base_url": config.DASHSCOPE_BASE_URL,
        "model": config.LLM_MODEL,
        "openai_package_version": openai.__version__,
        "proxy": {
            "HTTP_PROXY": safe_proxy("HTTP_PROXY"),
            "HTTPS_PROXY": safe_proxy("HTTPS_PROXY"),
            "ALL_PROXY": safe_proxy("ALL_PROXY"),
            "NO_PROXY_present": bool(os.getenv("NO_PROXY")),
            "system_proxies": safe_system_proxies(),
        },
        "dns": {},
        "direct_tls": {},
        "requests_https": {},
        "direct_requests_https": {},
        "minimal_sdk_call": {},
    }
    try:
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        result["dns"] = {"status": "PASS", "host": host, "address_count": len(addresses)}
    except Exception as exc:
        result["dns"] = {"status": "FAIL", "host": host, "exception_type": type(exc).__name__, "error": str(exc)}

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                result["direct_tls"] = {
                    "status": "PASS",
                    "tls_version": tls.version(),
                    "cipher": tls.cipher()[0] if tls.cipher() else None,
                }
    except Exception as exc:
        result["direct_tls"] = {"status": "FAIL", "exception_type": type(exc).__name__, "error": str(exc)}

    try:
        response = requests.get(config.DASHSCOPE_BASE_URL, timeout=15)
        result["requests_https"] = {
            "status": "PASS",
            "http_status": response.status_code,
            "reached_endpoint": True,
        }
    except Exception as exc:
        result["requests_https"] = {
            "status": "FAIL",
            "exception_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(config.DASHSCOPE_BASE_URL, timeout=15)
        result["direct_requests_https"] = {
            "status": "PASS",
            "http_status": response.status_code,
            "reached_endpoint": True,
        }
    except Exception as exc:
        result["direct_requests_https"] = {
            "status": "FAIL",
            "exception_type": type(exc).__name__,
            "error": str(exc),
        }

    if not config.DASHSCOPE_API_KEY.strip():
        result["minimal_sdk_call"] = {"status": "BLOCKED", "reason": "API key missing"}
    else:
        try:
            client = OpenAI(
                api_key=config.DASHSCOPE_API_KEY,
                base_url=config.DASHSCOPE_BASE_URL,
                timeout=20,
                max_retries=0,
                http_client=httpx.Client(trust_env=False, timeout=20),
            )
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": "你好"}],
                temperature=0,
                max_tokens=16,
            )
            content = completion.choices[0].message.content if completion.choices else None
            result["minimal_sdk_call"] = {
                "status": "PASS",
                "response_present": bool(content),
                "response_preview": content[:50] if content else None,
            }
            result["status"] = "PASS"
        except Exception as exc:
            cause = exc.__cause__
            result["minimal_sdk_call"] = {
                "status": "FAIL",
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "cause_type": type(cause).__name__ if cause else None,
                "cause": str(cause) if cause else None,
                "traceback": traceback.format_exc(limit=8),
            }

    output = ROOT / "RETAILRAG_QWEN_DIAGNOSTIC.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

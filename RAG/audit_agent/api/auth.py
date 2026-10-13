# auth.py
"""登录接口:登录获取 token"""
import requests

BASE_URL = "https://www.whhnhy.com:38868"
LOGIN_PATH = "/api/damp-app-api/agent/legal-agent/auth/loginNew"


def login(mobile: str, password: str) -> str:
    """登录,返回 accessToken"""
    resp = requests.post(
        f"{BASE_URL}{LOGIN_PATH}",
        json={"mobile": mobile, "password": password, "loginIdentity": "PERSONAL"},
        timeout=30,
    )
    return resp.json()["data"]["accessToken"]


if __name__ == "__main__":
    print(login("16610000003", "Aa123456"))

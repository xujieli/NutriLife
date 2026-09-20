"""用户与鉴权相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# 中国大陆手机号规则：1 开头，第二位为 3-9，共 11 位。
PHONE_RE = re.compile(r"^1[3-9]\d{9}$")


def is_valid_phone(phone: str) -> bool:
    """校验是否为合法的中国大陆手机号。"""
    return bool(PHONE_RE.match(phone))


class SendCodeRequest(BaseModel):
    """发送验证码请求体。"""

    phone: str = Field(description="中国大陆手机号")

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        if not is_valid_phone(value):
            raise ValueError("手机号格式不正确，请输入 11 位中国大陆手机号")
        return value


class LoginRequest(BaseModel):
    """手机号 + 验证码登录请求体。"""

    phone: str = Field(description="中国大陆手机号")
    code: str = Field(min_length=1, description="验证码（默认为手机号后 6 位）")

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        if not is_valid_phone(value):
            raise ValueError("手机号格式不正确，请输入 11 位中国大陆手机号")
        return value


class UserResponse(BaseModel):
    """用户信息响应体。"""

    id: str
    phone: str
    created_at: datetime
    last_login_at: datetime


class LoginResponse(BaseModel):
    """登录成功响应体。"""

    user: UserResponse

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dateutil import parser
from dotenv import load_dotenv

load_dotenv()


DATE_NAME_HINTS = {"date", "dt", "posting", "invoice", "bill", "transaction", "txn", "payment", "created", "purchase"}
AMOUNT_NAME_HINTS = {"amount", "amt", "value", "total", "price", "cost", "debit", "credit", "paid", "payable", "invoice", "net", "gross"}
VENDOR_NAME_HINTS = {"vendor", "supplier", "merchant", "payee", "seller", "company", "party", "client", "biller"}
CATEGORY_NAME_HINTS = {"category", "type", "class", "department", "gl", "head", "expense", "purpose"}
DESCRIPTION_NAME_HINTS = {"description", "details", "narration", "particular", "remarks", "memo", "note", "item"}

BUSINESS_ENTITY_WORDS = {
    "ltd", "limited", "pvt", "private", "inc", "llc", "corp", "corporation",
    "technologies", "systems", "solutions", "services", "india", "enterprises",
}

CATEGORY_KEYWORDS = {
    "software": "Software",
    "subscription": "Software",
    "cloud": "Cloud Services",
    "openai": "AI Services",
    "open ai": "AI Services",
    "claude": "AI Services",
    "anthropic": "AI Services",
    "amazon": "Marketplace / Cloud",
    "aws": "Cloud Services",
    "laptop": "Hardware",
    "laptops": "Hardware",
    "hardware": "Hardware",
    "hp": "Hardware",
    "printer": "Office Equipment",
    "office": "Office Supplies",
    "travel": "Travel",
    "flight": "Travel",
    "hotel": "Travel",
    "consulting": "Consulting",
    "consultancy": "Consulting",
    "training": "Training",
    "product": "Products",
    "products": "Products",
}

KNOWN_VENDORS = {
    "amazon": "Amazon",
    "aws": "Amazon Web Services",
    "open ai": "OpenAI",
    "openai": "OpenAI",
    "claude": "Claude",
    "anthropic": "Anthropic",
    "hp": "HP",
    "hp india": "HP India",
    "microsoft": "Microsoft",
    "google": "Google",
    "dell": "Dell",
    "lenovo": "Lenovo",
    "oracle": "Oracle",
    "salesforce": "Salesforce",
}


@dataclass
class ColumnDetection:
    column: str | None
    confidence: float
    reasons: list[str]


def _grok_api_key():
    key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
    if key:
        return key
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env_key:
                for name in ("XAI_API_KEY", "GROK_API_KEY", "GROQ_API_KEY"):
                    try:
                        value, _value_type = winreg.QueryValueEx(env_key, name)
                        if value:
                            return value
                    except FileNotFoundError:
                        pass
        except OSError:
            return None
    return None


def _ai_provider_config():
    if os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
        return {
            "name": "Grok",
            "url": "https://api.x.ai/v1/chat/completions",
            "model": os.getenv("XAI_MODEL", os.getenv("GROK_MODEL", "grok-4")),
        }
    if os.getenv("GROQ_API_KEY"):
        return {
            "name": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env_key:
                for name in ("XAI_API_KEY", "GROK_API_KEY"):
                    try:
                        value, _value_type = winreg.QueryValueEx(env_key, name)
                        if value:
                            return {
                                "name": "Grok",
                                "url": "https://api.x.ai/v1/chat/completions",
                                "model": os.getenv("XAI_MODEL", os.getenv("GROK_MODEL", "grok-4")),
                            }
                    except FileNotFoundError:
                        pass
                try:
                    value, _value_type = winreg.QueryValueEx(env_key, "GROQ_API_KEY")
                    if value:
                        return {
                            "name": "Groq",
                            "url": "https://api.groq.com/openai/v1/chat/completions",
                            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                        }
                except FileNotFoundError:
                    pass
        except OSError:
            pass
    return {
        "name": "Grok",
        "url": "https://api.x.ai/v1/chat/completions",
        "model": os.getenv("XAI_MODEL", os.getenv("GROK_MODEL", "grok-4")),
    }


def _call_grok(prompt: str, max_tokens: int = 900) -> str:
    api_key = _grok_api_key()
    if not api_key:
        raise RuntimeError("Grok/Groq API key is not configured")
    provider = _ai_provider_config()

    response = requests.post(
        provider["url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": provider["model"],
            "messages": [
                {"role": "system", "content": "You are Grok, a precise finance data mapping assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(name).lower()))


def _non_empty(series: pd.Series, limit: int = 250) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.head(limit)


def parse_money_value(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return None
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text):
        return None
    if re.fullmatch(r"\d{1,2}[-/]\d{2}", text):
        return None

    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = text.strip("()")
    text = text.replace("₹", "").replace("$", "").replace("rs.", "").replace("rs", "")
    text = text.replace("inr", "").replace("usd", "").replace(",", "").strip()
    text = re.sub(r"\s+", " ", text)

    multiplier = 1.0
    suffix_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(cr|crore|crores|l|lac|lakh|lakhs|k|thousand|m|million)\b", text)
    if suffix_match:
        number_text = suffix_match.group(1)
        suffix = suffix_match.group(2)
        if suffix in {"cr", "crore", "crores"}:
            multiplier = 10_000_000
        elif suffix in {"l", "lac", "lakh", "lakhs"}:
            multiplier = 100_000
        elif suffix in {"k", "thousand"}:
            multiplier = 1_000
        elif suffix in {"m", "million"}:
            multiplier = 1_000_000
    else:
        number_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not number_match:
            return None
        number_text = number_match.group(0)

    try:
        amount = float(number_text) * multiplier
    except ValueError:
        return None

    return -abs(amount) if negative else amount


def parse_date_value(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if 20000 <= numeric_value <= 70000:
            try:
                return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(numeric_value), unit="D")
            except (ValueError, OverflowError, TypeError):
                return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    if re.fullmatch(r"\d{5}", text):
        try:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(text), unit="D")
        except (ValueError, OverflowError, TypeError):
            return None

    if re.fullmatch(r"\d{1,2}[-/]\d{2}", text):
        first, second = re.split(r"[-/]", text)
        text = f"{first}-01-20{second}" if int(first) <= 12 else f"01-{first}-20{second}"

    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        try:
            parsed = parser.parse(text, yearfirst=True, fuzzy=False)
            return pd.Timestamp(parsed.date())
        except (ValueError, OverflowError, TypeError):
            return None

    for dayfirst in (True, False):
        try:
            parsed = parser.parse(text, dayfirst=dayfirst, fuzzy=False)
            if 1990 <= parsed.year <= 2100:
                return pd.Timestamp(parsed.date())
        except (ValueError, OverflowError, TypeError):
            pass
    return None


def _name_score(name: str, hints: set[str]) -> float:
    tokens = _tokens(name)
    joined = " ".join(tokens)
    hits = sum(1 for hint in hints if hint in tokens or hint in joined)
    return min(0.35, hits * 0.12)


def _detect_by_parser(df: pd.DataFrame, parser_func, hints: set[str], role: str) -> ColumnDetection:
    best = ColumnDetection(None, 0.0, [])
    for column in df.columns:
        values = _non_empty(df[column])
        if values.empty:
            continue

        parsed = values.map(parser_func)
        hit_ratio = parsed.notna().mean()
        name_bonus = _name_score(column, hints)
        confidence = min(0.99, hit_ratio * 0.75 + name_bonus)
        reasons = [f"{hit_ratio:.0%} sampled values parse as {role}"]
        if name_bonus:
            reasons.append("column name contains business hint")
        if confidence > best.confidence:
            best = ColumnDetection(column, float(round(confidence, 2)), reasons)

    if best.confidence < 0.35:
        best.column = None
    return best


def detect_date_column(df: pd.DataFrame) -> ColumnDetection:
    return _detect_by_parser(df, parse_date_value, DATE_NAME_HINTS, "dates")


def detect_amount_column(df: pd.DataFrame) -> ColumnDetection:
    return _detect_by_parser(df, parse_money_value, AMOUNT_NAME_HINTS, "money amounts")


def detect_vendor_column(df: pd.DataFrame) -> ColumnDetection:
    best = ColumnDetection(None, 0.0, [])
    for column in df.columns:
        values = _non_empty(df[column])
        if values.empty:
            continue

        text_ratio = values.map(lambda x: bool(re.search(r"[a-zA-Z]", x))).mean()
        unique_ratio = values.nunique(dropna=True) / len(values)
        joined_values = " ".join(values.head(80).str.lower().tolist())
        entity_hits = sum(1 for word in BUSINESS_ENTITY_WORDS | set(KNOWN_VENDORS) if word in joined_values)
        name_bonus = _name_score(column, VENDOR_NAME_HINTS)
        confidence = min(0.96, text_ratio * 0.35 + min(0.18, unique_ratio * 0.18) + min(0.25, entity_hits * 0.04) + name_bonus)
        reasons = [f"{text_ratio:.0%} sampled values contain text"]
        if entity_hits:
            reasons.append("supplier or company-like words detected")
        if name_bonus:
            reasons.append("column name contains vendor hint")
        if confidence > best.confidence:
            best = ColumnDetection(column, float(round(confidence, 2)), reasons)

    if best.confidence < 0.32:
        best.column = None
    return best


def detect_category_column(df: pd.DataFrame) -> ColumnDetection:
    best = ColumnDetection(None, 0.0, [])
    for column in df.columns:
        values = _non_empty(df[column])
        if values.empty:
            continue

        text_ratio = values.map(lambda x: bool(re.search(r"[a-zA-Z]", x))).mean()
        unique_ratio = values.nunique(dropna=True) / len(values)
        keyword_hits = values.str.lower().map(lambda x: any(keyword in x for keyword in CATEGORY_KEYWORDS)).mean()
        name_bonus = _name_score(column, CATEGORY_NAME_HINTS)
        confidence = min(0.95, text_ratio * 0.25 + (1 - unique_ratio) * 0.25 + keyword_hits * 0.3 + name_bonus)
        reasons = [f"{keyword_hits:.0%} sampled values contain category keywords"]
        if unique_ratio < 0.55:
            reasons.append("values repeat like business labels")
        if name_bonus:
            reasons.append("column name contains category hint")
        if confidence > best.confidence:
            best = ColumnDetection(column, float(round(confidence, 2)), reasons)

    if best.confidence < 0.30:
        best.column = None
    return best


def detect_description_column(df: pd.DataFrame) -> ColumnDetection:
    best = ColumnDetection(None, 0.0, [])
    for column in df.columns:
        values = _non_empty(df[column])
        if values.empty:
            continue

        avg_len = values.str.len().mean()
        wordy_ratio = values.map(lambda x: len(re.findall(r"[a-zA-Z]+", x)) >= 3).mean()
        name_bonus = _name_score(column, DESCRIPTION_NAME_HINTS)
        confidence = min(0.92, wordy_ratio * 0.45 + min(0.25, avg_len / 160) + name_bonus)
        reasons = [f"{wordy_ratio:.0%} sampled values look like sentence text"]
        if name_bonus:
            reasons.append("column name contains description hint")
        if confidence > best.confidence:
            best = ColumnDetection(column, float(round(confidence, 2)), reasons)

    if best.confidence < 0.28:
        best.column = None
    return best


def infer_vendor_from_text(text: Any) -> str | None:
    value = str(text or "").lower()
    for key, label in KNOWN_VENDORS.items():
        if re.search(rf"\b{re.escape(key)}\b", value):
            return label

    company_match = re.search(
        r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,3})\s+"
        r"(?:pvt|private|ltd|limited|inc|corp|corporation|technologies|systems|solutions|services)\b",
        str(text or ""),
    )
    if company_match:
        return company_match.group(0).strip()
    return None


def infer_category_from_text(text: Any) -> str:
    value = str(text or "").lower()
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in value:
            return category
    return "Uncategorized"


def _json_from_text(text: str) -> Any:
    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ai_schema_fallback(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not _grok_api_key():
        schema["ai_status"] = {
            "column": None,
            "confidence": 0.0,
            "reasons": ["Grok disabled: XAI_API_KEY is not configured"],
        }
        return schema

    sample_df = df.head(12).copy()
    sample_payload = {
        "columns": list(df.columns.astype(str)),
        "detected_schema": schema,
        "sample_rows": sample_df.fillna("").astype(str).to_dict(orient="records"),
    }
    prompt = f"""
You are a finance schema detection assistant for fraud analytics.
Map uploaded CSV columns to these roles: date, amount, vendor, category, description.
Use column names, sample value formats, repeated values, and business meaning.
Return only valid JSON with this exact shape:
{{
  "date": {{"column": "column name or null", "confidence": 0.0, "reason": "short reason"}},
  "amount": {{"column": "column name or null", "confidence": 0.0, "reason": "short reason"}},
  "vendor": {{"column": "column name or null", "confidence": 0.0, "reason": "short reason"}},
  "category": {{"column": "column name or null", "confidence": 0.0, "reason": "short reason"}},
  "description": {{"column": "column name or null", "confidence": 0.0, "reason": "short reason"}}
}}

CSV analysis input:
{json.dumps(sample_payload, ensure_ascii=False)}
"""
    try:
        text = _call_grok(prompt, max_tokens=900)
        ai_result = _json_from_text(text)
    except Exception as exc:
        schema["ai_status"] = {
            "column": None,
            "confidence": 0.0,
            "reasons": [f"Grok fallback failed: {exc.__class__.__name__}"],
        }
        return schema

    if not isinstance(ai_result, dict):
        schema["ai_status"] = {
            "column": None,
            "confidence": 0.0,
            "reasons": ["Grok fallback returned an unreadable response"],
        }
        return schema

    valid_columns = set(df.columns.astype(str))
    for role in ["date", "amount", "vendor", "category", "description"]:
        suggestion = ai_result.get(role, {})
        column = suggestion.get("column") if isinstance(suggestion, dict) else None
        confidence = float(suggestion.get("confidence", 0)) if isinstance(suggestion, dict) else 0.0
        reason = suggestion.get("reason", "Grok semantic mapping") if isinstance(suggestion, dict) else "Grok semantic mapping"
        if column in valid_columns and confidence > schema.get(role, {}).get("confidence", 0):
            schema[role] = {
                "column": column,
                "confidence": min(0.98, round(confidence, 2)),
                "reasons": [f"Grok assisted: {reason}"],
            }

    schema["ai_status"] = {
        "column": "Grok",
        "confidence": 1.0,
        "reasons": ["Grok reviewed schema mappings"],
    }
    return schema


def ai_extract_entities(descriptions: list[str]) -> dict[str, dict[str, str]]:
    if not _grok_api_key() or not descriptions:
        return {}

    limited_descriptions = descriptions[:30]
    prompt = f"""
Extract vendor and category for finance transaction descriptions.
Return only valid JSON object where each key is the original description and value is:
{{"vendor": "vendor name or Unknown Vendor", "category": "business category"}}
Use categories such as Hardware, Software, Cloud Services, Office Supplies, Travel, Consulting, Products, AI Services.

Descriptions:
{json.dumps(limited_descriptions, ensure_ascii=False)}
"""
    try:
        text = _call_grok(prompt, max_tokens=1200)
        parsed = _json_from_text(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def detect_schema(df: pd.DataFrame, use_claude: bool = True) -> dict[str, dict[str, Any]]:
    detections = {
        "date": detect_date_column(df),
        "amount": detect_amount_column(df),
        "vendor": detect_vendor_column(df),
        "category": detect_category_column(df),
        "description": detect_description_column(df),
    }
    schema = {
        role: {"column": detection.column, "confidence": detection.confidence, "reasons": detection.reasons}
        for role, detection in detections.items()
    }
    if use_claude:
        schema = ai_schema_fallback(df, schema)
    else:
        schema["ai_status"] = {
            "column": None,
            "confidence": 0.0,
            "reasons": ["Grok disabled by user toggle"],
        }
    return schema


def normalize_transactions(df: pd.DataFrame, mapping: dict[str, str | None] | None = None, use_claude: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    working = df.copy()
    working.columns = working.columns.astype(str).str.strip()
    schema = detect_schema(working, use_claude=use_claude)
    selected = {role: data["column"] for role, data in schema.items()}
    if mapping:
        selected.update({role: column for role, column in mapping.items() if column})

    normalized = working.copy()
    date_col = selected.get("date")
    amount_col = selected.get("amount")
    vendor_col = selected.get("vendor")
    category_col = selected.get("category")
    description_col = selected.get("description")

    normalized["date"] = working[date_col].map(parse_date_value) if date_col else pd.NaT
    normalized["amount"] = working[amount_col].map(parse_money_value) if amount_col else None
    normalized["description"] = (
        working[description_col].astype(str).str.strip()
        if description_col
        else working.apply(lambda row: " | ".join(str(value) for value in row.dropna().head(4)), axis=1)
    )

    if vendor_col and vendor_col != description_col:
        normalized["vendor"] = working[vendor_col].astype(str).str.strip()
    else:
        normalized["vendor"] = normalized["description"].map(infer_vendor_from_text).fillna("Unknown Vendor")

    inferred_vendor = normalized["description"].map(infer_vendor_from_text)
    normalized["vendor"] = normalized["vendor"].mask(
        normalized["vendor"].astype(str).str.lower().isin({"", "nan", "none", "unknown", "unknown vendor"}),
        inferred_vendor,
    ).fillna("Unknown Vendor")

    normalized["category"] = (
        working[category_col].astype(str).str.strip()
        if category_col and category_col != description_col
        else normalized["description"].map(infer_category_from_text)
    )
    normalized["category"] = normalized["category"].replace({"": "Uncategorized", "nan": "Uncategorized"}).fillna("Uncategorized")

    if use_claude:
        needs_ai = normalized[
            normalized["vendor"].eq("Unknown Vendor") | normalized["category"].eq("Uncategorized")
        ]["description"].dropna().astype(str).drop_duplicates().head(30).tolist()
        ai_entities = ai_extract_entities(needs_ai)
        if ai_entities:
            normalized["ai_vendor_suggestion"] = normalized["description"].map(
                lambda text: ai_entities.get(str(text), {}).get("vendor")
            )
            normalized["ai_category_suggestion"] = normalized["description"].map(
                lambda text: ai_entities.get(str(text), {}).get("category")
            )
            normalized["vendor"] = normalized["vendor"].mask(
                normalized["vendor"].eq("Unknown Vendor") & normalized["ai_vendor_suggestion"].notna(),
                normalized["ai_vendor_suggestion"],
            )
            normalized["category"] = normalized["category"].mask(
                normalized["category"].eq("Uncategorized") & normalized["ai_category_suggestion"].notna(),
                normalized["ai_category_suggestion"],
            )
        else:
            normalized["ai_vendor_suggestion"] = ""
            normalized["ai_category_suggestion"] = ""

    normalized["source_date_column"] = date_col or ""
    normalized["source_amount_column"] = amount_col or ""
    normalized["source_vendor_column"] = vendor_col or ""
    normalized["source_category_column"] = category_col or ""
    normalized["source_description_column"] = description_col or ""

    normalized = normalized.dropna(subset=["amount"]).copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.sort_values(by=["date", "amount"], na_position="last")
    return normalized, schema

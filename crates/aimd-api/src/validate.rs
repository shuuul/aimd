//! Request-body validation with FastAPI/pydantic-compatible 422 bodies.
//!
//! Ports the lax-mode coercions pydantic applies to `ProcessRequest`
//! (`"3"` -> 3, `"yes"` -> true, ...) and the `{"detail": [...]}` error
//! items (`type`, `loc`, `msg`, `input`, `ctx`).

use std::fmt;

use aimd_core::TaskType;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::schemas::ProcessRequest;

/// JSON object that keeps key order (the workspace `serde_json` sorts
/// map keys; pydantic echoes inputs in request order).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderedObject(pub Vec<(String, Value)>);

impl OrderedObject {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl<'de> Deserialize<'de> for OrderedObject {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OrderedObject;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<OrderedObject, A::Error> {
                let mut out: Vec<(String, Value)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Value>()? {
                    // json.loads keeps the first position and the last value.
                    if let Some(slot) = out.iter_mut().find(|(key, _)| *key == k) {
                        slot.1 = v;
                    } else {
                        out.push((k, v));
                    }
                }
                Ok(OrderedObject(out))
            }
        }
        d.deserialize_map(V)
    }
}

impl Serialize for OrderedObject {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// One `loc` element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum LocPart {
    Key(String),
    Index(usize),
}

impl From<&str> for LocPart {
    fn from(value: &str) -> Self {
        LocPart::Key(value.to_string())
    }
}

/// Echoed `input` value.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Echo {
    Value(Value),
    Object(OrderedObject),
}

/// One pydantic validation error item.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ErrorItem {
    #[serde(rename = "type")]
    pub kind: String,
    pub loc: Vec<LocPart>,
    pub msg: String,
    pub input: Echo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Value>,
}

impl ErrorItem {
    pub fn new(kind: &str, loc: Vec<LocPart>, msg: &str, input: Value) -> Self {
        Self {
            kind: kind.to_string(),
            loc,
            msg: msg.to_string(),
            input: Echo::Value(input),
            ctx: None,
        }
    }

    pub fn with_ctx(mut self, ctx: Value) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// `{"type":"missing","loc":loc,"msg":"Field required","input":null}`.
    pub fn missing(loc: Vec<LocPart>) -> Self {
        Self::new("missing", loc, "Field required", Value::Null)
    }
}

/// `{"detail": [items]}` body.
pub fn detail_body(items: &[ErrorItem]) -> String {
    #[derive(Serialize)]
    struct Body<'a> {
        detail: &'a [ErrorItem],
    }
    serde_json::to_string(&Body { detail: items }).unwrap_or_else(|_| "{}".to_string())
}

/// FastAPI's JSON content-type test (`application/json` or `+json`).
fn is_json_content_type(content_type: Option<&str>) -> bool {
    let Some(ct) = content_type else {
        return true;
    };
    let essence = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match essence.split_once('/') {
        Some(("application", sub)) => sub == "json" || sub.ends_with("+json"),
        _ => false,
    }
}

/// Python `json` decode message for a serde_json error (approximate).
fn python_json_message(err: &serde_json::Error) -> &'static str {
    let text = err.to_string();
    if text.starts_with("key must be a string") {
        "Expecting property name enclosed in double quotes"
    } else if text.starts_with("expected `,` or `}`") || text.starts_with("expected `,` or `]`") {
        "Expecting ',' delimiter"
    } else if text.starts_with("expected `:`") {
        "Expecting ':' delimiter"
    } else if text.starts_with("trailing characters") {
        "Extra data"
    } else if text.starts_with("EOF while parsing a string") {
        "Unterminated string starting at"
    } else if text.starts_with("invalid escape") {
        "Invalid \\escape"
    } else if text.starts_with("control character") {
        "Invalid control character at"
    } else {
        "Expecting value"
    }
}

/// Character offset of a serde_json error (Python `JSONDecodeError.pos`).
fn error_position(text: &str, err: &serde_json::Error) -> usize {
    if err.is_eof() {
        return text.chars().count();
    }
    let (line, column) = (err.line(), err.column());
    let mut offset = 0usize;
    for (index, l) in text.split('\n').enumerate() {
        if index + 1 == line {
            let byte_col = column.saturating_sub(1).min(l.len());
            let mut end = byte_col;
            while !l.is_char_boundary(end) {
                end -= 1;
            }
            return offset + l[..end].chars().count();
        }
        offset += l.chars().count() + 1;
    }
    offset
}

/// Decode a FastAPI JSON body: `Ok(None)` for an empty body.
fn decode_body(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<Option<Result<OrderedObject, Value>>, Box<ErrorItem>> {
    if body.is_empty() {
        return Ok(None);
    }
    if !is_json_content_type(content_type) {
        return Ok(Some(Err(Value::String(
            String::from_utf8_lossy(body).into_owned(),
        ))));
    }
    let text = String::from_utf8_lossy(body);
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Err(Box::new(
                ErrorItem::new(
                    "json_invalid",
                    vec!["body".into(), LocPart::Index(error_position(&text, &e))],
                    "JSON decode error",
                    json!({}),
                )
                .with_ctx(json!({"error": python_json_message(&e)})),
            ));
        }
    };
    if value.is_object() {
        let ordered: OrderedObject =
            serde_json::from_str(&text).unwrap_or_else(|_| OrderedObject::default());
        Ok(Some(Ok(ordered)))
    } else {
        Ok(Some(Err(value)))
    }
}

fn field_loc(name: &str) -> Vec<LocPart> {
    vec!["body".into(), name.into()]
}

fn opt_str(obj: &OrderedObject, name: &str, errors: &mut Vec<ErrorItem>) -> Option<String> {
    match obj.get(name) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push(ErrorItem::new(
                "string_type",
                field_loc(name),
                "Input should be a valid string",
                other.clone(),
            ));
            None
        }
    }
}

/// Pydantic lax `bool`.
pub fn coerce_bool(value: &Value) -> Result<bool, (&'static str, &'static str)> {
    const PARSING: (&str, &str) = (
        "bool_parsing",
        "Input should be a valid boolean, unable to interpret input",
    );
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) => match n.as_f64() {
            Some(0.0) => Ok(false),
            Some(1.0) => Ok(true),
            _ => Err(PARSING),
        },
        Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
            _ => Err(PARSING),
        },
        _ => Err(("bool_type", "Input should be a valid boolean")),
    }
}

fn bool_field(obj: &OrderedObject, name: &str, default: bool, errors: &mut Vec<ErrorItem>) -> bool {
    match obj.get(name) {
        None => default,
        Some(v) => coerce_bool(v).unwrap_or_else(|(kind, msg)| {
            errors.push(ErrorItem::new(kind, field_loc(name), msg, v.clone()));
            default
        }),
    }
}

const INT_PARSING: (&str, &str) = (
    "int_parsing",
    "Input should be a valid integer, unable to parse string as an integer",
);
const INT_SIZE: (&str, &str) = (
    "int_parsing_size",
    "Unable to parse input string as an integer, exceeded maximum size",
);

fn float_to_int(f: f64) -> Result<i64, (&'static str, &'static str)> {
    if !f.is_finite() {
        return Err(("finite_number", "Input should be a finite number"));
    }
    if f.fract() != 0.0 {
        return Err((
            "int_from_float",
            "Input should be a valid integer, got a number with a fractional part",
        ));
    }
    if f.abs() >= 9.223_372_036_854_775e18 {
        return Err(INT_SIZE);
    }
    Ok(f as i64)
}

/// Pydantic lax `int` from a string (whitespace-trimmed; `"3.0"` -> 3).
pub fn coerce_int_str(s: &str) -> Result<i64, (&'static str, &'static str)> {
    let t = s.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        let cleaned: String = t.chars().filter(|&c| c != '_').collect();
        return cleaned.parse::<i64>().map_err(|_| INT_SIZE);
    }
    match t.parse::<f64>() {
        Ok(f) if f.is_finite() && f.fract() == 0.0 => float_to_int(f).map_err(|_| INT_SIZE),
        _ => Err(INT_PARSING),
    }
}

/// Pydantic lax `int`.
pub fn coerce_int(value: &Value) -> Result<i64, (&'static str, &'static str)> {
    match value {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if n.is_u64() {
                Err(INT_SIZE)
            } else {
                float_to_int(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => coerce_int_str(s),
        _ => Err(("int_type", "Input should be a valid integer")),
    }
}

fn opt_int(obj: &OrderedObject, name: &str, errors: &mut Vec<ErrorItem>) -> Option<i64> {
    match obj.get(name) {
        None | Some(Value::Null) => None,
        Some(v) => match coerce_int(v) {
            Ok(i) => Some(i),
            Err((kind, msg)) => {
                errors.push(ErrorItem::new(kind, field_loc(name), msg, v.clone()));
                None
            }
        },
    }
}

fn opt_task(obj: &OrderedObject, errors: &mut Vec<ErrorItem>) -> Option<TaskType> {
    const EXPECTED: &str = "'transcript', 'convert' or 'ocr'";
    match obj.get("task_type") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let parsed = v.as_str().and_then(TaskType::parse);
            if parsed.is_none() {
                errors.push(
                    ErrorItem::new(
                        "literal_error",
                        field_loc("task_type"),
                        &format!("Input should be {EXPECTED}"),
                        v.clone(),
                    )
                    .with_ctx(json!({ "expected": EXPECTED })),
                );
            }
            parsed
        }
    }
}

/// Parse a `ProcessRequest` body exactly like FastAPI + pydantic.
pub fn parse_process_request(
    content_type: Option<&str>,
    body: &[u8],
) -> Result<ProcessRequest, Vec<ErrorItem>> {
    let obj = match decode_body(content_type, body) {
        Err(item) => return Err(vec![*item]),
        Ok(None) => return Err(vec![ErrorItem::missing(vec!["body".into()])]),
        Ok(Some(Err(value))) => {
            return Err(vec![ErrorItem::new(
                "model_attributes_type",
                vec!["body".into()],
                "Input should be a valid dictionary or object to extract fields from",
                value,
            )]);
        }
        Ok(Some(Ok(obj))) => obj,
    };
    validate_process_object(&obj)
}

/// Field + model validation of a decoded request object.
pub fn validate_process_object(obj: &OrderedObject) -> Result<ProcessRequest, Vec<ErrorItem>> {
    let mut errors = Vec::new();
    let e = &mut errors;
    let request = ProcessRequest {
        input_source: opt_str(obj, "input_source", e),
        blob_id: opt_str(obj, "blob_id", e),
        task_type: opt_task(obj, e),
        output_file: opt_str(obj, "output_file", e),
        model: opt_str(obj, "model", e),
        precision: opt_str(obj, "precision", e),
        asr_base_url: opt_str(obj, "asr_base_url", e),
        asr_model: opt_str(obj, "asr_model", e),
        asr_api_key: opt_str(obj, "asr_api_key", e),
        ocr_base_url: opt_str(obj, "ocr_base_url", e),
        ocr_model: opt_str(obj, "ocr_model", e),
        ocr_api_key: opt_str(obj, "ocr_api_key", e),
        context: opt_str(obj, "context", e),
        metadata_context: bool_field(obj, "metadata_context", true, e),
        language: opt_str(obj, "language", e),
        start: opt_int(obj, "start", e),
        end: opt_int(obj, "end", e),
        save_original: opt_str(obj, "save_original", e),
        cookies: opt_str(obj, "cookies", e),
        cookies_from_browser: opt_str(obj, "cookies_from_browser", e),
        raw_transcript: bool_field(obj, "raw_transcript", false, e),
    };
    if !errors.is_empty() {
        return Err(errors);
    }
    let has_input = request
        .input_source
        .as_deref()
        .is_some_and(|s| !s.is_empty());
    if has_input == request.has_blob() {
        return Err(vec![ErrorItem {
            kind: "value_error".into(),
            loc: vec!["body".into()],
            msg: "Value error, Exactly one of input_source or blob_id must be set".into(),
            input: Echo::Object(obj.clone()),
            ctx: Some(json!({"error": {}})),
        }]);
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(body: &str) -> String {
        detail_body(&parse_process_request(Some("application/json"), body.as_bytes()).unwrap_err())
    }

    #[test]
    fn error_bodies_match_fastapi() {
        // Captured from the Python API (tests/test_api_blobs.py scenarios + probes).
        assert_eq!(
            err("{}"),
            r#"{"detail":[{"type":"value_error","loc":["body"],"msg":"Value error, Exactly one of input_source or blob_id must be set","input":{},"ctx":{"error":{}}}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":"a","blob_id":"b"}"#),
            r#"{"detail":[{"type":"value_error","loc":["body"],"msg":"Value error, Exactly one of input_source or blob_id must be set","input":{"input_source":"a","blob_id":"b"},"ctx":{"error":{}}}]}"#
        );
        assert_eq!(
            err(r#"{"a":"#),
            r#"{"detail":[{"type":"json_invalid","loc":["body",5],"msg":"JSON decode error","input":{},"ctx":{"error":"Expecting value"}}]}"#
        );
        assert_eq!(
            err("[]"),
            r#"{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"Input should be a valid dictionary or object to extract fields from","input":[]}]}"#
        );
        assert_eq!(
            detail_body(&parse_process_request(None, b"").unwrap_err()),
            r#"{"detail":[{"type":"missing","loc":["body"],"msg":"Field required","input":null}]}"#
        );
        assert_eq!(
            detail_body(
                &parse_process_request(Some("text/plain"), br#"{"input_source":"x"}"#).unwrap_err()
            ),
            r#"{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"Input should be a valid dictionary or object to extract fields from","input":"{\"input_source\":\"x\"}"}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":5,"task_type":"x"}"#),
            r#"{"detail":[{"type":"string_type","loc":["body","input_source"],"msg":"Input should be a valid string","input":5},{"type":"literal_error","loc":["body","task_type"],"msg":"Input should be 'transcript', 'convert' or 'ocr'","input":"x","ctx":{"expected":"'transcript', 'convert' or 'ocr'"}}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":"a","start":"abc"}"#),
            r#"{"detail":[{"type":"int_parsing","loc":["body","start"],"msg":"Input should be a valid integer, unable to parse string as an integer","input":"abc"}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":"a","start":1.5}"#),
            r#"{"detail":[{"type":"int_from_float","loc":["body","start"],"msg":"Input should be a valid integer, got a number with a fractional part","input":1.5}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":"a","metadata_context":"maybe","raw_transcript":2}"#),
            r#"{"detail":[{"type":"bool_parsing","loc":["body","metadata_context"],"msg":"Input should be a valid boolean, unable to interpret input","input":"maybe"},{"type":"bool_parsing","loc":["body","raw_transcript"],"msg":"Input should be a valid boolean, unable to interpret input","input":2}]}"#
        );
        assert_eq!(
            err(r#"{"input_source":"a","metadata_context":null}"#),
            r#"{"detail":[{"type":"bool_type","loc":["body","metadata_context"],"msg":"Input should be a valid boolean","input":null}]}"#
        );
    }

    #[test]
    fn lax_coercions_match_pydantic() {
        let r = parse_process_request(
            None,
            br#"{"input_source":"a","start":" 3 ","end":"4.0","metadata_context":"YES","raw_transcript":1.0,"extra":1}"#,
        )
        .unwrap();
        assert_eq!((r.start, r.end), (Some(3), Some(4)));
        assert!(r.metadata_context && r.raw_transcript);
        let r = parse_process_request(None, br#"{"input_source":"a","start":true}"#).unwrap();
        assert_eq!(r.start, Some(1));
        let r = parse_process_request(None, br#"{"input_source":"a","blob_id":""}"#).unwrap();
        assert_eq!(r.blob_id.as_deref(), Some(""));
        assert!(parse_process_request(None, br#"{"input_source":""}"#).is_err());
        let r =
            parse_process_request(Some("application/vnd.api+json"), br#"{"blob_id":"x"}"#).unwrap();
        assert!(r.metadata_context);
    }
}

//! OpenAPI document. `openapi.json` is the schema FastAPI generated for the
//! Python API (frozen wire contract); the version, the native-engine help
//! texts, and the bearer security scheme are patched in at startup.

use aimd_core::output::{CONTEXT_HELP_TEXT, MODEL_HELP_TEXT, PRECISION_HELP_TEXT};
use serde_json::{Value, json};

const FROZEN: &str = include_str!("openapi.json");

/// Serialized OpenAPI document (`bearer` adds `bearerAuth` security).
pub fn document(bearer: bool) -> String {
    let mut schema: Value = serde_json::from_str(FROZEN).expect("embedded openapi.json is valid");
    schema["info"]["version"] = json!(env!("CARGO_PKG_VERSION"));
    let props = &mut schema["components"]["schemas"]["ProcessRequest"]["properties"];
    for (field, text) in [
        ("model", MODEL_HELP_TEXT),
        ("precision", PRECISION_HELP_TEXT),
        ("context", CONTEXT_HELP_TEXT),
    ] {
        if props[field].is_object() {
            props[field]["description"] = json!(text);
        }
    }
    if bearer {
        schema["components"]["securitySchemes"] =
            json!({"bearerAuth": {"type": "http", "scheme": "bearer"}});
        schema["security"] = json!([{"bearerAuth": []}]);
    }
    schema.to_string()
}

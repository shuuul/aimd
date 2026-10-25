//! MCP stdio server for aimd (ports `aimd.interfaces.mcp.app`).
//!
//! Tools: `healthz` and `process_input`. The tool schemas are hand-written to
//! match the FastMCP-generated schemas of the Python server byte for byte;
//! arguments are validated here (pydantic-style messages) so the wire
//! behavior is the same as the Python server's.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aimd_core::output::{PersistError, get_request_temp_dir, persist_result_output_if_requested};
use aimd_core::{NoHooks, ProcessInput, ProcessResult, TaskType};
use aimd_process::{Runtime, process_input_with};
use rmcp::ErrorData;
use rmcp::RoleServer;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use serde_json::{Map, Value, json};

/// Docstring of the Python `process_input` tool (kept verbatim, including its
/// indentation, because FastMCP publishes it as the tool description).
const PROCESS_INPUT_DESCRIPTION: &str = "Process audio/video/url/documents and return markdown context.\n\n    precision selects model quantization: 4bit, 6bit, 8bit, or bf16 (dash\n    variants like 4-bit are accepted). macOS MLX backends select the matching\n    mlx-community checkpoint (default 4bit when omitted). CUDA Transformers\n    backends only accept bf16 (requires CUDA bf16 support) and otherwise keep\n    automatic dtype selection when precision is omitted.\n\n    context is free-form ASR biasing text (proper nouns, names, domain\n    vocabulary) that improves transcription accuracy. metadata_context\n    (default True) additionally injects URL page metadata (title,\n    description, tags) as ASR context for URL inputs.\n    ";

/// A tool call failure; rendered as an `isError` tool result.
#[derive(Debug)]
pub struct ToolFailure(pub String);

/// The aimd MCP server. The runtime is injected so tests can mock it.
#[derive(Clone)]
pub struct AimdMcp {
    runtime: Arc<dyn Runtime>,
}

impl AimdMcp {
    pub fn new(runtime: Arc<dyn Runtime>) -> Self {
        Self { runtime }
    }

    /// Dispatch a tool call by name; the success value is the structured
    /// content. Errors carry the text published as the tool error.
    pub async fn call(
        &self,
        name: &str,
        arguments: Map<String, Value>,
    ) -> Result<Value, ToolFailure> {
        match name {
            "healthz" => Ok(json!({"status": "ok"})),
            "process_input" => self.process_input(arguments).await,
            other => Err(ToolFailure(format!("Unknown tool: {other}"))),
        }
    }

    async fn process_input(&self, arguments: Map<String, Value>) -> Result<Value, ToolFailure> {
        let args = ProcessArgs::parse(&arguments)
            .map_err(|msg| ToolFailure(format!("Error executing tool process_input: {msg}")))?;
        let runtime = Arc::clone(&self.runtime);
        let joined = tokio::task::spawn_blocking(move || run_process_input(args, runtime.as_ref()))
            .await
            .map_err(|e| ToolFailure(format!("Error executing tool process_input: {e}")))?;
        joined.map_err(|message| {
            ToolFailure(format!("Error executing tool process_input: {message}"))
        })
    }
}

/// Validated `process_input` arguments (the Python signature).
struct ProcessArgs {
    request: ProcessInput,
    output_file: Option<String>,
}

impl ProcessArgs {
    fn parse(arguments: &Map<String, Value>) -> Result<Self, String> {
        let mut errors: Vec<(String, String)> = Vec::new();
        let mut field = Fields {
            arguments,
            errors: &mut errors,
        };
        let input_source = field.required_string("input_source");
        let task_type = field.task_type();
        let model = field.optional_string("model");
        let language = field.optional_string("language");
        let start = field.optional_int("start");
        let end = field.optional_int("end");
        let output_file = field.optional_string("output_file");
        let save_original = field.optional_string("save_original");
        let cookies = field.optional_string("cookies");
        let cookies_from_browser = field.optional_string("cookies_from_browser");
        let raw_transcript = field.bool_or("raw_transcript", false);
        let precision = field.optional_string("precision");
        let context = field.optional_string("context");
        let metadata_context = field.bool_or("metadata_context", true);
        let asr_base_url = field.optional_string("asr_base_url");
        let asr_model = field.optional_string("asr_model");
        let asr_api_key = field.optional_string("asr_api_key");
        let ocr_base_url = field.optional_string("ocr_base_url");
        let ocr_model = field.optional_string("ocr_model");
        let ocr_api_key = field.optional_string("ocr_api_key");
        if !errors.is_empty() {
            let mut text = format!(
                "{} validation error{} for process_inputArguments",
                errors.len(),
                if errors.len() == 1 { "" } else { "s" }
            );
            for (name, message) in &errors {
                text.push_str(&format!("\n{name}\n  {message}"));
            }
            return Err(text);
        }
        let non_empty_path = |v: Option<String>| v.filter(|s| !s.is_empty()).map(PathBuf::from);
        let request = ProcessInput {
            input_source: input_source.unwrap_or_default(),
            task_type,
            model,
            language,
            start,
            end,
            save_original: non_empty_path(save_original),
            cookies: non_empty_path(cookies),
            cookies_from_browser,
            temp_dir: None,
            raw_transcript,
            precision,
            context,
            metadata_context,
            asr_base_url,
            asr_model,
            asr_api_key,
            ocr_base_url,
            ocr_model,
            ocr_api_key,
        };
        Ok(Self {
            request,
            output_file,
        })
    }
}

/// Pydantic-flavored field extraction over the raw JSON arguments.
struct Fields<'a> {
    arguments: &'a Map<String, Value>,
    errors: &'a mut Vec<(String, String)>,
}

impl Fields<'_> {
    fn get(&self, name: &str) -> Option<&Value> {
        self.arguments.get(name)
    }

    fn fail(&mut self, name: &str, message: &str) {
        self.errors.push((name.to_string(), message.to_string()));
    }

    fn required_string(&mut self, name: &str) -> Option<String> {
        match self.get(name) {
            None => {
                self.fail(name, "Field required [type=missing]");
                None
            }
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                self.fail(name, "Input should be a valid string [type=string_type]");
                None
            }
        }
    }

    fn optional_string(&mut self, name: &str) -> Option<String> {
        match self.get(name) {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => {
                self.fail(name, "Input should be a valid string [type=string_type]");
                None
            }
        }
    }

    fn optional_int(&mut self, name: &str) -> Option<i64> {
        match self.get(name) {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) if n.is_i64() => n.as_i64(),
            Some(Value::Number(n)) if n.as_f64().is_some_and(|f| f.fract() == 0.0) => {
                n.as_f64().map(|f| f as i64)
            }
            Some(_) => {
                self.fail(name, "Input should be a valid integer [type=int_type]");
                None
            }
        }
    }

    fn bool_or(&mut self, name: &str, default: bool) -> bool {
        match self.get(name) {
            None => default,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                self.fail(name, "Input should be a valid boolean [type=bool_type]");
                default
            }
        }
    }

    fn task_type(&mut self) -> Option<TaskType> {
        match self.get("task_type") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if TaskType::parse(s).is_some() => TaskType::parse(s),
            Some(other) => {
                let (shown, kind) = match other {
                    Value::String(s) => (format!("'{s}'"), "str"),
                    Value::Bool(b) => (b.to_string(), "bool"),
                    Value::Number(n) => (n.to_string(), if n.is_f64() { "float" } else { "int" }),
                    v => (v.to_string(), "other"),
                };
                self.fail(
                    "task_type",
                    &format!(
                        "Input should be 'transcript', 'convert' or 'ocr' \
                         [type=literal_error, input_value={shown}, input_type={kind}]"
                    ),
                );
                None
            }
        }
    }
}

/// The blocking body: temp dir, core processing, persistence, payload.
fn run_process_input(args: ProcessArgs, runtime: &dyn Runtime) -> Result<Value, String> {
    let ProcessArgs {
        mut request,
        output_file,
    } = args;
    request.temp_dir = get_request_temp_dir().map_err(|e| e.to_string())?;
    let result = process_input_with(&request, &NoHooks, runtime).map_err(|e| e.to_string())?;
    let persisted =
        persist_result_output_if_requested(&result, output_file.as_deref().map(Path::new))
            .map_err(|e| match e {
                PersistError::Aimd(e) => e.to_string(),
                PersistError::Io(e) => e.to_string(),
            })?;
    if persisted.ignored_output_file {
        aimd_core::log::emit(
            aimd_core::log::Level::Warning,
            module_path!(),
            "Ignoring output_file for document asset conversions; output is a directory.",
        );
    }
    Ok(result_payload(
        &result,
        persisted.output_file,
        persisted.output_dir,
    ))
}

/// Same keys and key order as the Python `_process_result_payload`.
fn result_payload(
    result: &ProcessResult,
    output_file: Option<String>,
    output_dir: Option<String>,
) -> Value {
    // serde_json maps are key-sorted; `to_ordered_text` restores Python order.
    json!({
        "task_type": result.task_type.as_str(),
        "title": result.text_context.title,
        "markdown": result.markdown,
        "asset_base_uri": result.asset_base_uri,
        "chunk_list": result.text_context.chunk_list,
        "split_header_level": result.text_context.split_header_level,
        "platform": result.platform,
        "output_file": output_file,
        "output_dir": output_dir,
    })
}

/// Python-ordered keys of the `process_input` payload.
const PAYLOAD_KEY_ORDER: [&str; 9] = [
    "task_type",
    "title",
    "markdown",
    "asset_base_uri",
    "chunk_list",
    "split_header_level",
    "platform",
    "output_file",
    "output_dir",
];

/// Indented (2 spaces) JSON text with the payload keys in Python order, as
/// FastMCP renders the text content block.
fn to_ordered_text(value: &Value) -> String {
    let Value::Object(map) = value else {
        return serde_json::to_string_pretty(value).unwrap_or_default();
    };
    let mut keys: Vec<&str> = PAYLOAD_KEY_ORDER
        .iter()
        .copied()
        .filter(|k| map.contains_key(*k))
        .collect();
    keys.extend(
        map.keys()
            .map(String::as_str)
            .filter(|k| !PAYLOAD_KEY_ORDER.contains(k)),
    );
    if keys.is_empty() {
        return "{}".to_string();
    }
    let mut out = String::from("{\n");
    for (i, key) in keys.iter().enumerate() {
        let rendered = serde_json::to_string_pretty(&map[*key]).unwrap_or_default();
        let indented = rendered.replace('\n', "\n  ");
        out.push_str(&format!(
            "  {}: {}{}\n",
            serde_json::to_string(key).unwrap_or_default(),
            indented,
            if i + 1 < keys.len() { "," } else { "" }
        ));
    }
    out.push('}');
    out
}

fn schema(value: Value) -> Arc<Map<String, Value>> {
    match value {
        Value::Object(map) => Arc::new(map),
        _ => unreachable!("tool schemas are objects"),
    }
}

fn nullable(inner: Value, title: &str) -> Value {
    json!({"anyOf": [inner, {"type": "null"}], "default": null, "title": title})
}

/// The two tool definitions, matching FastMCP's generated schemas.
pub fn tools() -> Vec<Tool> {
    let string = || json!({"type": "string"});
    let integer = || json!({"type": "integer"});
    let healthz = Tool::new(
        "healthz",
        "Health check for aimd MCP server.",
        schema(json!({
            "properties": {}, "title": "healthzArguments", "type": "object"
        })),
    )
    .with_raw_output_schema(schema(json!({
        "additionalProperties": {"type": "string"},
        "title": "healthzDictOutput",
        "type": "object"
    })));
    let strings = [
        ("asr_api_key", "Asr Api Key"),
        ("asr_base_url", "Asr Base Url"),
        ("asr_model", "Asr Model"),
        ("context", "Context"),
        ("cookies", "Cookies"),
        ("cookies_from_browser", "Cookies From Browser"),
        ("language", "Language"),
        ("model", "Model"),
        ("ocr_api_key", "Ocr Api Key"),
        ("ocr_base_url", "Ocr Base Url"),
        ("ocr_model", "Ocr Model"),
        ("output_file", "Output File"),
        ("precision", "Precision"),
        ("save_original", "Save Original"),
    ];
    let mut props = Map::new();
    for (name, title) in strings {
        props.insert(name.into(), nullable(string(), title));
    }
    props.insert("end".into(), nullable(integer(), "End"));
    props.insert("start".into(), nullable(integer(), "Start"));
    props.insert(
        "input_source".into(),
        json!({"title": "Input Source", "type": "string"}),
    );
    props.insert(
        "metadata_context".into(),
        json!({"default": true, "title": "Metadata Context", "type": "boolean"}),
    );
    props.insert(
        "raw_transcript".into(),
        json!({"default": false, "title": "Raw Transcript", "type": "boolean"}),
    );
    props.insert(
        "task_type".into(),
        nullable(
            json!({"enum": ["transcript", "convert", "ocr"], "type": "string"}),
            "Task Type",
        ),
    );
    let process = Tool::new(
        "process_input",
        PROCESS_INPUT_DESCRIPTION,
        schema(json!({
            "properties": props,
            "required": ["input_source"],
            "title": "process_inputArguments",
            "type": "object"
        })),
    )
    .with_raw_output_schema(schema(json!({
        "additionalProperties": true,
        "title": "process_inputDictOutput",
        "type": "object"
    })));
    vec![healthz, process]
}

impl ServerHandler for AimdMcp {
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new("aimd", "");
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            tools: tools(),
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let name: Cow<'static, str> = request.name;
        let arguments = request.arguments.unwrap_or_default();
        let result = match self.call(&name, arguments).await {
            Ok(value) => {
                let mut result =
                    CallToolResult::success(vec![ContentBlock::text(to_ordered_text(&value))]);
                result.structured_content = Some(value);
                result
            }
            Err(ToolFailure(message)) => CallToolResult::error(vec![ContentBlock::text(message)]),
        };
        Ok(result.into())
    }
}

#[cfg(test)]
mod tests;

//! `aimd` CLI (WS-16). Ports `aimd.interfaces.cli`: hand-written argument
//! parsing over the frozen Typer flag surface, stderr logging, and the
//! Python output-path, message, and exit-code contract.

mod app;
mod args;

fn main() {
    app::load_dotenv(std::path::Path::new(".env"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = app::run(
        &args,
        &|key| std::env::var(key).ok(),
        &aimd_process::NativeRuntime,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}

#[cfg(test)]
mod tests;

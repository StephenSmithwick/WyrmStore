pub mod config;
pub mod daemon;
pub mod embed;
mod store;
pub mod types;

use crate::config::Config;
use crate::daemon::Daemon;
use crate::types::DaemonRequest;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::thread;

pub fn run() -> io::Result<()> {
    let config = Config::from_env()?;
    if let Some(socket_path) = config.socket_path.clone() {
        return run_socket(config, socket_path);
    }

    let mut daemon = Daemon::new(config.clone())?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

    handle_stream(&mut daemon, config.framing, &mut reader, &mut writer)?;

    Ok(())
}

fn read_framed_json<R: BufRead>(reader: &mut R) -> io::Result<String> {
    let mut buf = String::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "<<<END>>>" {
            break;
        }
        buf.push_str(trimmed);
    }
    Ok(buf)
}

fn run_socket(config: Config, socket_path: std::path::PathBuf) -> io::Result<()> {
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }
    let listener = UnixListener::bind(&socket_path)?;
    let daemon = Arc::new(Mutex::new(Daemon::new(config.clone())?));
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("socket accept failed: {err}");
                continue;
            }
        };
        let daemon = Arc::clone(&daemon);
        let framing = config.framing;
        thread::spawn(move || {
            if let Err(err) = handle_socket_connection(daemon, framing, stream) {
                eprintln!("socket connection error: {err}");
            }
        });
    }
    Ok(())
}

fn handle_socket_connection(
    daemon: Arc<Mutex<Daemon>>,
    framing: bool,
    stream: UnixStream,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let json = if line == "<<<BEGIN>>>" {
            read_framed_json(&mut reader)?
        } else {
            line.to_string()
        };
        if json.trim().is_empty() {
            continue;
        }

        let request: DaemonRequest = match serde_json::from_str(&json) {
            Ok(req) => req,
            Err(err) => {
                eprintln!("failed to parse request: {err}");
                continue;
            }
        };

        let response = {
            let mut daemon = daemon.lock().map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "daemon lock poisoned")
            })?;
            daemon.handle_request(request)
        };

        match response {
            Ok(response) => {
                let out = serde_json::to_string(&response)?;
                if framing {
                    writeln!(writer, "<<<BEGIN>>>")?;
                    writeln!(writer, "{out}")?;
                    writeln!(writer, "<<<END>>>")?;
                } else {
                    writeln!(writer, "{out}")?;
                }
                writer.flush()?;
            }
            Err(err) => {
                eprintln!("request failed: {err}");
            }
        }
    }

    Ok(())
}

fn handle_stream<R: BufRead, W: Write>(
    daemon: &mut Daemon,
    framing: bool,
    reader: &mut R,
    writer: &mut W,
) -> io::Result<()> {
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let json = if line == "<<<BEGIN>>>" {
            read_framed_json(reader)?
        } else {
            line.to_string()
        };
        if json.trim().is_empty() {
            continue;
        }

        let request: DaemonRequest = match serde_json::from_str(&json) {
            Ok(req) => req,
            Err(err) => {
                eprintln!("failed to parse request: {err}");
                continue;
            }
        };

        match daemon.handle_request(request) {
            Ok(response) => {
                let out = serde_json::to_string(&response)?;
                if framing {
                    writeln!(writer, "<<<BEGIN>>>")?;
                    writeln!(writer, "{out}")?;
                    writeln!(writer, "<<<END>>>")?;
                } else {
                    writeln!(writer, "{out}")?;
                }
                writer.flush()?;
            }
            Err(err) => {
                eprintln!("request failed: {err}");
            }
        }
    }
    Ok(())
}

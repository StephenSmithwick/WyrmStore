#![cfg(not(feature = "mini-test"))]

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use uuid::Uuid;

fn spawn_daemon(data_dir: &Path) -> std::process::Child {
    spawn_daemon_with_env(data_dir, &[])
}

fn spawn_daemon_with_env(data_dir: &Path, extra_env: &[(&str, &str)]) -> std::process::Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_memoryd"));
    cmd.env("MEMD_CONFIG_PATH", "config/memoryd.test.toml")
        .env("MEMD_DATA_DIR", data_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.spawn().expect("failed to spawn daemon")
}

fn read_response(reader: &mut BufReader<std::process::ChildStdout>) -> serde_json::Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("read response");
    assert!(!line.trim().is_empty(), "empty response line");
    serde_json::from_str(&line).expect("parse response json")
}

fn temp_dir() -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("memoryd_fastembed_test_{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn e2e_fastembed_fixture() {
    let dir = temp_dir();
    let fixture = fs::read_to_string("tests/fixtures/fastembed.jsonl").expect("read fixture");
    let lines: Vec<&str> = fixture.lines().collect();

    let mut child = spawn_daemon(&dir);
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    for (idx, line) in lines.iter().enumerate() {
        writeln!(stdin, "{line}").expect("write input");
        stdin.flush().expect("flush input");
        let response = read_response(&mut reader);
        if idx == lines.len() - 1 {
            let injected = response["memory_out"]["injected_chunks"]
                .as_array()
                .expect("injected_chunks array");
            assert!(!injected.is_empty(), "expected injected chunks");

            let messages = response["messages"].as_array().expect("messages array");
            let developer = messages
                .iter()
                .find(|m| m["role"] == "developer")
                .expect("developer message");
            let content = developer["content"].as_str().unwrap_or("");
            assert!(content.contains("PostgreSQL"), "expected memory content");
        }
    }

    drop(stdin);
    let _ = child.wait();
}

// Load test lives in tests/e2e_load.rs to keep normal e2e runs lighter.

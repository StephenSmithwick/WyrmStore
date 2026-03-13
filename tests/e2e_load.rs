#![cfg(feature = "load-test")]
#![cfg(not(feature = "mini-test"))]

use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use uuid::Uuid;

fn spawn_daemon_with_env(data_dir: &Path, extra_env: &[(&str, &str)]) -> std::process::Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_memoryd"));
    cmd.env("MEMD_CONFIG_PATH", "config/memoryd.prod.toml")
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

fn send_request(
    stdin: &mut std::process::ChildStdin,
    reader: &mut BufReader<std::process::ChildStdout>,
    value: serde_json::Value,
) -> serde_json::Value {
    let line = value.to_string();
    writeln!(stdin, "{line}").expect("write input");
    stdin.flush().expect("flush input");
    read_response(reader)
}

fn temp_dir() -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("memoryd_fastembed_load_{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn e2e_fastembed_load_test_multiple_segments() {
    let dir = temp_dir();
    let mut child = spawn_daemon_with_env(
        &dir,
        &[
            ("MEMD_ACTIVE_MAX_ENTRIES", "8"),
            ("MEMD_SOFT_TOKENS", "10"),
            ("MEMD_HARD_TOKENS", "12"),
            ("MEMD_HNSW_EF_SEARCH", "16"),
        ],
    );
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    let mut messages = vec![json!({"role":"system","content":"You are a helpful assistant."})];
    for i in 0..60usize {
        messages.push(json!({"role":"user","content":format!("Load test message {i} with padding tokens to trigger compaction.")}));
        messages.push(json!({"role":"assistant","content":"Ack."}));
        let request = json!({
            "messages": messages,
            "memory_in": { "topic_id": "load" }
        });
        let _ = send_request(&mut stdin, &mut reader, request);
    }

    let segments_dir = dir.join("load").join("segments");
    let mut sealed = 0usize;
    if let Ok(entries) = std::fs::read_dir(&segments_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("seg_") && name.ends_with(".bin"))
                .unwrap_or(false)
            {
                sealed += 1;
            }
        }
    }

    assert!(
        sealed >= 3,
        "expected multiple sealed segments, found {sealed}"
    );

    drop(stdin);
    let _ = child.wait();
}

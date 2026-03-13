use std::fs;
use uuid::Uuid;

mod support;

fn build_config(temp: &std::path::Path) -> memoryd::config::Config {
    let mut config =
        memoryd::config::Config::from_toml_path("config/memoryd.mini.toml")
            .expect("load mini config");
    config.data_dir = temp.to_path_buf();
    config
}

fn temp_dir() -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("memoryd_test_{}", Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn system_message() -> memoryd::types::Message {
    memoryd::types::Message {
        role: "system".to_string(),
        content: "You are a helpful assistant.".to_string(),
        tool_calls: None,
        extra: Default::default(),
    }
}

fn user_message(content: &str) -> memoryd::types::Message {
    memoryd::types::Message {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        extra: Default::default(),
    }
}

fn assistant_message(content: &str) -> memoryd::types::Message {
    memoryd::types::Message {
        role: "assistant".to_string(),
        content: content.to_string(),
        tool_calls: None,
        extra: Default::default(),
    }
}

fn make_request(messages: Vec<memoryd::types::Message>) -> memoryd::types::DaemonRequest {
    memoryd::types::DaemonRequest {
        messages,
        memory_in: Some(memoryd::types::MemoryIn {
            topic_id: Some("proj".to_string()),
            related_topic_ids: None,
            corrections: None,
        }),
    }
}

#[test]
fn e2e_basic_fixture() {
    let dir = temp_dir();
    let fixture = fs::read_to_string("tests/fixtures/basic.jsonl").expect("read fixture");
    let lines: Vec<&str> = fixture.lines().collect();

    let config = build_config(&dir);
    let embedder = Box::new(support::hash_embed::HashEmbedder::new(64));
    let mut daemon = memoryd::daemon::Daemon::new_with_embedder(config, embedder);

    for (idx, line) in lines.iter().enumerate() {
        let request: memoryd::types::DaemonRequest =
            serde_json::from_str(line).expect("parse request");
        let response = daemon.handle_request(request).expect("handle request");
        let response = serde_json::to_value(response).expect("serialize response");
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
}

#[test]
fn e2e_corrections_fixture() {
    let dir = temp_dir();
    let fixture = fs::read_to_string("tests/fixtures/corrections_template.jsonl").expect("read fixture");
    let lines: Vec<&str> = fixture.lines().collect();

    let config = build_config(&dir);
    let embedder = Box::new(support::hash_embed::HashEmbedder::new(64));
    let mut daemon = memoryd::daemon::Daemon::new_with_embedder(config, embedder);

    // Line 1
    let request: memoryd::types::DaemonRequest =
        serde_json::from_str(lines[0]).expect("parse request");
    let _ = daemon.handle_request(request).expect("handle request");

    // Line 2 - capture chunk id
    let request: memoryd::types::DaemonRequest =
        serde_json::from_str(lines[1]).expect("parse request");
    let response = daemon.handle_request(request).expect("handle request");
    let response = serde_json::to_value(response).expect("serialize response");
    let injected = response["memory_out"]["injected_chunks"]
        .as_array()
        .expect("injected_chunks array");
    assert!(!injected.is_empty(), "expected injected chunks");
    let chunk_id = injected[0]["id"].as_str().expect("chunk id");

    // Line 3 - corrections
    let correction_line = lines[2].replace("{{CHUNK_ID}}", chunk_id);
    let request: memoryd::types::DaemonRequest =
        serde_json::from_str(&correction_line).expect("parse request");
    let response = daemon.handle_request(request).expect("handle request");
    let response = serde_json::to_value(response).expect("serialize response");
    if let Some(signals) = response["memory_out"]["signals"].as_array() {
        let has_failed = signals.iter().any(|s| s["type"] == "correction_failed");
        assert!(!has_failed, "unexpected correction_failed signal");
    }

    // Line 4 - query after update
    let request: memoryd::types::DaemonRequest =
        serde_json::from_str(lines[3]).expect("parse request");
    let response = daemon.handle_request(request).expect("handle request");
    let response = serde_json::to_value(response).expect("serialize response");
    let messages = response["messages"].as_array().expect("messages array");
    let developer = messages
        .iter()
        .find(|m| m["role"] == "developer")
        .expect("developer message");
    let content = developer["content"].as_str().unwrap_or("");
    assert!(content.contains("SQLite"), "expected updated memory content");
    assert!(
        !content.contains("PostgreSQL"),
        "expected old memory to be deprecated"
    );
}

#[test]
fn e2e_sealed_segment_retrieval() {
    let dir = temp_dir();
    let mut config = build_config(&dir);
    config.active_max_entries = 1;
    config.soft_tokens = 6;
    config.hard_tokens = 8;

    let embedder = Box::new(support::hash_embed::HashEmbedder::new(64));
    let mut daemon = memoryd::daemon::Daemon::new_with_embedder(config, embedder);

    let mut messages = vec![system_message()];
    messages.push(user_message(
        "Alpha key is 123. Alpha key is 123. Alpha key is 123.",
    ));
    messages.push(assistant_message("Noted."));
    messages.push(user_message("Beta key is 456."));

    let request = make_request(messages.clone());
    let _ = daemon.handle_request(request).expect("handle request");

    let segments_dir = dir.join("proj").join("segments");
    let seg_bin = segments_dir.join("seg_0001.bin");
    let seg_graph = segments_dir.join("seg_0001.hnsw.graph");
    let seg_data = segments_dir.join("seg_0001.hnsw.data");
    assert!(seg_bin.exists(), "expected sealed segment bin");
    assert!(seg_graph.exists(), "expected hnsw graph");
    assert!(seg_data.exists(), "expected hnsw data");

    let active_contents =
        fs::read_to_string(dir.join("proj").join("active.bin")).unwrap_or_default();
    assert!(
        !active_contents.contains("Alpha key is 123"),
        "expected alpha key to be sealed"
    );

    messages.push(assistant_message("Ok."));
    messages.push(user_message("What is the alpha key?"));
    let request = make_request(messages);
    let response = daemon.handle_request(request).expect("handle request");
    let response = serde_json::to_value(response).expect("serialize response");
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
    assert!(
        content.contains("Alpha key is 123"),
        "expected sealed segment memory in injection"
    );
}

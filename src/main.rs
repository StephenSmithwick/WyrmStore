fn main() {
    if let Err(err) = memoryd::run() {
        eprintln!("fatal: {err}");
        std::process::exit(1);
    }
}

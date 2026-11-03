//! Prompts from 127.0.0.1:8765 (one line per connection, live_protocol) and from the window's prompt box. Of those
//! waiting only the newest is taken: a prompt is encoded between two chunks, and one typed while T5 was busy with
//! another makes the one before it pointless.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use live_protocol::Cut;
use std::path::Path;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub cut: Cut,
    pub text: String,
}

/// The newest prompt not taken yet.
#[derive(Clone, Default)]
pub struct Inbox(Arc<Mutex<Option<Prompt>>>);

impl Inbox {
    pub fn put(&self, line: &str) -> bool {
        match live_protocol::parse(line) {
            Some((cut, text)) => {
                *self.0.lock().unwrap() = Some(Prompt { cut, text });
                true
            }
            None => false,
        }
    }

    pub fn take(&self) -> Option<Prompt> {
        self.0.lock().unwrap().take()
    }

    /// Takes the lines of every connection to `port` into the inbox, in a thread of its own.
    pub fn listen(&self, port: u16) -> std::io::Result<()> {
        let server = TcpListener::bind(("127.0.0.1", port))?;
        let inbox = self.clone();
        std::thread::Builder::new().name("prompts".into()).spawn(move || {
            for connection in server.incoming().flatten() {
                let mut line = String::new();
                if BufReader::new(connection).read_line(&mut line).is_ok() && inbox.put(&line) {
                    println!("prompt received: {}", line.trim());
                }
            }
        })?;
        Ok(())
    }
}

/// Sends one prompt to a running Live (live_video.py --send).
pub fn send(cut: Cut, prompt: &str, port: u16) -> std::io::Result<()> {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port))?;
    s.write_all(format!("{}\n", live_protocol::line(cut, prompt)).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_waiting_prompt_is_taken_once() {
        let inbox = Inbox::default();
        assert!(inbox.put("a fox"));
        assert!(inbox.put("/scene a market at night"));
        assert!(!inbox.put("   "));
        assert_eq!(inbox.take(), Some(Prompt { cut: Cut::Scene, text: "a market at night".into() }));
        assert_eq!(inbox.take(), None);
    }

    #[test]
    fn a_line_over_the_socket_reaches_the_inbox() {
        let inbox = Inbox::default();
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        inbox.listen(port).unwrap();
        send(Cut::New, "a red fox", port).unwrap();
        for _ in 0..100 {
            if let Some(p) = inbox.take() {
                assert_eq!(p, Prompt { cut: Cut::New, text: "a red fox".into() });
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the prompt did not arrive");
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub cut: Cut,
    pub text: String,
    pub chunks: usize,
}

/// The steps of `chain`, each prompt written out whole; `chunks` a prompt by default, `root` the repository.
pub fn chain_steps(chain: &Value, chunks: usize, root: &Path) -> Result<Vec<Step>, String> {
    let chunks = chain["chunks"].as_u64().map_or(chunks, |c| c as usize);
    if let Some(reference) = chain["ref"].as_str() {
        let (file, line) = reference.rsplit_once('#').ok_or(format!("{reference}: no #line"))?;
        let text = std::fs::read_to_string(root.join(file)).map_err(|e| format!("{file}: {e}"))?;
        let line: usize = line.parse().map_err(|_| format!("{reference}: the line is not a number"))?;
        if line == 0 { return Err("reference lines start at 1".into()); }
        let row: Value = serde_json::from_str(text.lines().nth(line - 1).ok_or(format!("{reference}: no such line"))?)
            .map_err(|e| format!("{reference}: {e}"))?;
        let prompts = row["prompts"].as_array().ok_or(format!("{reference}: no prompts"))?;
        return Ok(prompts.iter().enumerate()
            .map(|(i, p)| Step { cut: if i == 0 { Cut::New } else { Cut::Same }, text: p.as_str().unwrap_or("").into(), chunks })
            .collect());
    }
    let (mut card, mut light) = (None::<String>, None::<String>);
    let mut steps = Vec::new();
    for (i, s) in chain["steps"].as_array().ok_or("a chain has steps or a ref")?.iter().enumerate() {
        let cut = match s["cut"].as_str().unwrap_or(if i == 0 { "new" } else { "same" }) {
            "new" => Cut::New,
            "same" => Cut::Same,
            "scene" => Cut::Scene,
            other => return Err(format!("unknown cut {other}")),
        };
        let text = if let Some(t) = s["text"].as_str() {
            t.to_string()
        } else {
            if let Some(c) = s["card"].as_str() { card = Some(c.into()); }
            if let Some(l) = s["light"].as_str() { light = Some(l.into()); }
            let camera: Vec<&str> = [chain["camera"].as_str(), light.as_deref()].into_iter().flatten().collect();
            let action = s["action"].as_str().ok_or(format!("step {i}: no text and no action"))?;
            format!("{} {action} {}.", card.as_deref().ok_or(format!("step {i}: no card"))?, camera.join(", "))
        };
        steps.push(Step { cut, text, chunks: s["chunks"].as_u64().map_or(chunks, |c| c as usize) });
    }
    Ok(steps)
}

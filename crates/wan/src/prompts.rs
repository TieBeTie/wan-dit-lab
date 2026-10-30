//! Prompts from 127.0.0.1:8765 (one line per connection, live_protocol) and from the window's prompt box. Of those
//! waiting only the newest is taken: a prompt is encoded between two chunks, and one typed while T5 was busy with
//! another makes the one before it pointless.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use live_protocol::Cut;

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

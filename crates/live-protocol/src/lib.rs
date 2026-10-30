//! The line a prompt crosses to the Live video in: one UTF-8 line per TCP connection to 127.0.0.1:8765. Plain text
//! goes on with the scene; "/scene " is LongLive 2.0's cut to a new shot that still sees the old one; "/new " starts
//! a new video, the cache emptied. The voice tutor writes it (tutor-core), Live reads it.

pub const PORT: u16 = 8765;

/// How a prompt comes into the video.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// The scene goes on under the new prompt.
    Same,
    /// LongLive 2.0's cut: a new shot of the same world.
    Scene,
    /// A new video.
    New,
}

const SCENE: &str = "/scene";
const NEW: &str = "/new";

/// The line of `prompt` under `cut`, its whitespace collapsed to single spaces.
pub fn line(cut: Cut, prompt: &str) -> String {
    let head = match cut {
        Cut::Same => "",
        Cut::Scene => "/scene ",
        Cut::New => "/new ",
    };
    format!("{head}{}", prompt.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// The cut and the prompt of a line; None for an empty prompt.
pub fn parse(line: &str) -> Option<(Cut, String)> {
    let line = line.trim();
    // a command is a whole word: "/newspaper" is a prompt (live_video.py took it for "/new" and "spaper")
    let command = |name: &str| line.strip_prefix(name).filter(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace));
    let (cut, rest) = if let Some(rest) = command(SCENE) {
        (Cut::Scene, rest)
    } else if let Some(rest) = command(NEW) {
        (Cut::New, rest)
    } else {
        (Cut::Same, line)
    };
    let prompt = rest.split_whitespace().collect::<Vec<_>>().join(" ");
    (!prompt.is_empty()).then_some((cut, prompt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_line_per_prompt_with_its_cut() {
        assert_eq!(line(Cut::Same, " a fox\n walks "), "a fox walks");
        assert_eq!(line(Cut::New, "winter"), "/new winter");
        assert_eq!(line(Cut::Scene, "a shot"), "/scene a shot");
    }

    #[test]
    fn parse_reads_what_line_writes() {
        for cut in [Cut::Same, Cut::Scene, Cut::New] {
            assert_eq!(parse(&line(cut, "a red fox  in snow")), Some((cut, "a red fox in snow".into())));
        }
        assert_eq!(parse("  /new   "), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn a_command_is_a_whole_word() {
        assert_eq!(parse("/scene a busy night market"), Some((Cut::Scene, "a busy night market".into())));
        assert_eq!(parse("/newspaper on a table"), Some((Cut::Same, "/newspaper on a table".into())));
    }
}

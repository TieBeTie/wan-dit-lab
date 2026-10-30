//! T5 on a thread of its own: a prompt is encoded (its blocks read from the disk, about 1.6 s) while the video goes on
//! with the prompt it has, and the new one is taken between two chunks once it is ready.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;

use crate::gpu::{DeviceBuffer, Event, Gpu, Stream};
use crate::prompts::Prompt;
use crate::t5::T5;
use crate::video::Result;

/// A prompt to encode, the text as the model takes it, and the event after which the video has read the embeddings
/// of the one before (they are written over).
type Request = (Prompt, String, Event);

pub struct Encoder {
    requests: Sender<Request>,
    done: Receiver<std::result::Result<Prompt, String>>,
    embeddings: Arc<DeviceBuffer>,
    shape: [i64; 3],
    busy: bool,
    /// the newest prompt that came while another was encoded
    waiting: Option<(Prompt, String)>,
}

impl Encoder {
    pub fn spawn(mut t5: T5, embeddings: DeviceBuffer) -> Result<Encoder> {
        let (requests, inbox) = channel::<Request>();
        let (finished, done) = channel();
        let embeddings = Arc::new(embeddings);
        let shape = t5.shape();
        let out = embeddings.clone();
        std::thread::Builder::new().name("t5".into()).spawn(move || {
            let gpu = match Gpu::new().and_then(|g| g.bind().map(|_| g)) {
                Ok(g) => g,
                Err(e) => return drop(finished.send(Err(e.to_string()))),
            };
            let stream = match Stream::new() {
                Ok(s) => s,
                Err(e) => return drop(finished.send(Err(e.to_string()))),
            };
            for (prompt, text, read) in inbox {
                let r = read.synchronize().map_err(|e| e.to_string())
                    .and_then(|_| t5.encode(&text, &out, &stream).map_err(|e| e.to_string()));
                if finished.send(r.map(|_| prompt)).is_err() {
                    break;
                }
            }
            drop(gpu);
        })?;
        Ok(Encoder { requests, done, embeddings, shape, busy: false, waiting: None })
    }

    /// Encodes `prompt` (as the model takes it, `text`) after the one being encoded; a newer one replaces it while it
    /// waits. `read`: the video has read the last embeddings after it (recorded on its stream).
    pub fn encode(&mut self, prompt: Prompt, text: String, read: Event) -> Result<()> {
        if self.busy {
            self.waiting = Some((prompt, text));
            return Ok(());
        }
        self.requests.send((prompt, text, read)).map_err(|_| "T5's thread is gone")?;
        self.busy = true;
        Ok(())
    }

    /// The prompt whose embeddings are ready, if one is; `wait`: until it is.
    pub fn ready(&mut self, wait: bool) -> Result<Option<Prompt>> {
        if !self.busy {
            return Ok(None);
        }
        let r = if wait {
            self.done.recv().map_err(|_| "T5's thread is gone")?
        } else {
            match self.done.try_recv() {
                Ok(r) => r,
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => return Err("T5's thread is gone".into()),
            }
        };
        self.busy = false;
        Ok(Some(r?))
    }

    /// The prompt that came while another was encoded, to be sent once the video has read that one's embeddings.
    pub fn take_waiting(&mut self) -> Option<(Prompt, String)> {
        self.waiting.take()
    }

    pub fn tensor(&self) -> aoti::Tensor {
        self.embeddings.tensor(0, aoti::DType::F32, &self.shape)
    }
}

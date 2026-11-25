//! The GPU queue, transparent to the caller: one card, one run at a time. Every wan.exe that is about to use the GPU
//! (any run with a config, bench, check, render, export) first takes a ticket in artifacts/queue/tickets/<id>.json.
//! When the card is free and no ticket is ahead, it runs at once; otherwise it prints
//! "[очередь] поставлено <id>: позиция N, ждёт" and waits, then prints the start and, at the end, the exit code.
//! The command is always the same `wan --config X`; copies of the runner (wan-*.exe) share the queue.
//!
//!     wan queue status          running and waiting tickets, the last ended ones, GPU runs outside the queue
//!     wan queue cancel <id>     a waiting ticket ends as cancelled and its wan.exe exits
//!
//! Ticket order is the order of the ids (UTC time of submission). The card is busy while artifacts/queue/gpu.lock is
//! held, or while a GPU process that holds no waiting ticket is alive: wan*.exe, live.exe, a Python script of
//! scripts/experiments|conversion|calibration|generation (runs started outside the queue, the old Python queue).
//! A ticket of a dead process ends as interrupted. Tickets are never deleted.

use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::video::Result;

const ENDED: [&str; 4] = ["done", "failed", "cancelled", "interrupted"];

#[derive(Clone, Debug, PartialEq)]
pub struct Proc {
    pub pid: u32,
    /// the process that started it (0 when unknown): a run never waits for its own ancestors
    pub parent: u32,
    pub name: String,
    pub cmdline: String,
}

pub fn dir() -> PathBuf { crate::root().join("artifacts").join("queue") }

/// A run that holds the card: it ends its ticket with `finish`, and the lock goes with it.
pub struct Ticket {
    queue: PathBuf,
    pub id: String,
    _lock: File,
}

/// The processes of `procs` that use the GPU, leaving out the pids in `own` and every process above them (a driver
/// that started a run is its parent, not a GPU job beside it).
pub fn gpu_processes(procs: &[Proc], own: &HashSet<u32>) -> Vec<Proc> {
    const SCRIPTS: [&str; 4] = ["experiments", "conversion", "calibration", "generation"];
    let parent: std::collections::HashMap<u32, u32> = procs.iter().map(|p| (p.pid, p.parent)).collect();
    let mut mine = own.clone();
    for &start in own {
        let mut at = start;
        while let Some(&up) = parent.get(&at) {
            if up == 0 || !mine.insert(up) { break; }
            at = up;
        }
    }
    procs.iter().filter(|p| !mine.contains(&p.pid)).filter(|p| {
        let name = p.name.to_lowercase();
        let line = p.cmdline.to_lowercase().replace('/', "\\");
        (name.starts_with("wan") && name.ends_with(".exe")) || name == "live.exe"
            || SCRIPTS.iter().any(|d| line.contains(&format!("scripts\\{d}\\")))
    }).cloned().collect()
}

/// (ISO time, ticket id) of now, in UTC; ids sort in the order of submission.
fn stamp() -> (String, String) {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let (secs, nanos) = (t.as_secs() as i64, t.subsec_nanos());
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil date from days since 1970-01-01 (H. Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let (hh, mm, ss) = (rem / 3600, rem % 3600 / 60, rem % 60);
    (format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z"), format!("{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}-{nanos:09}"))
}

fn path_of(queue: &Path, id: &str) -> PathBuf { queue.join("tickets").join(format!("{id}.json")) }

fn write(queue: &Path, ticket: &Value) -> Result<()> {
    let path = path_of(queue, ticket["id"].as_str().ok_or("ticket without id")?);
    fs::create_dir_all(path.parent().unwrap())?;
    let tmp = path.with_extension(format!("{}.tmp", stamp().1));
    fs::write(&tmp, serde_json::to_string_pretty(ticket)?)?;
    let mut last = None;
    for attempt in 0..50 { // a reader may hold the file open for a moment on Windows
        match fs::rename(&tmp, &path) {
            Ok(()) => return Ok(()),
            Err(e) => { last = Some(e); std::thread::sleep(Duration::from_millis(10 * (attempt + 1))); }
        }
    }
    Err(last.unwrap().into())
}

fn read(path: &Path) -> Option<Value> {
    for attempt in 0..50 {
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str(&text) { return Some(v); }
        }
        std::thread::sleep(Duration::from_millis(10 * (attempt + 1)));
    }
    None
}

fn update(queue: &Path, id: &str, fields: Value) -> Result<Value> {
    let mut t = read(&path_of(queue, id)).ok_or_else(|| format!("no ticket {id}"))?;
    for (k, v) in fields.as_object().unwrap() { t[k] = v.clone(); }
    write(queue, &t)?;
    Ok(t)
}

/// Every ticket, oldest first.
pub fn tickets(queue: &Path) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(queue.join("tickets")) else { return vec![] };
    let mut all: Vec<Value> = entries.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json")).filter_map(|p| read(&p)).collect();
    all.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    all
}

fn status(t: &Value) -> &str { t["status"].as_str().unwrap_or("") }
fn pid(t: &Value) -> u32 { t["pid"].as_u64().unwrap_or(0) as u32 }

fn try_lock(queue: &Path) -> Result<Option<File>> {
    fs::create_dir_all(queue)?;
    let file = File::options().read(true).write(true).create(true).truncate(false).open(queue.join("gpu.lock"))?;
    Ok(file.try_lock().is_ok().then_some(file))
}

/// Take a ticket and wait until it is this run's turn: no ticket ahead, the lock free, no GPU process outside.
pub fn take_turn(queue: &Path, me: u32, command: &[String], config: Option<&Path>,
                 procs: &mut dyn FnMut() -> Vec<Proc>, say: &mut dyn FnMut(&str), poll: Duration) -> Result<Ticket> {
    let (submitted, id) = stamp();
    let cwd = std::env::current_dir().map(|d| d.display().to_string()).unwrap_or_default();
    write(queue, &json!({"id": id, "status": "queued", "config": config.map(|c| c.display().to_string()),
        "command": command, "cwd": cwd, "submitted": submitted, "pid": me}))?;
    let mut last = String::new();
    loop {
        // tickets first, then the process list: a run that took its ticket after this read is not judged by an older list
        let mut all = tickets(queue);
        let ps = procs();
        let alive: HashSet<u32> = ps.iter().map(|p| p.pid).chain([me]).collect();
        for t in all.iter_mut().filter(|t| matches!(status(t), "queued" | "running") && !alive.contains(&pid(t))) {
            update(queue, t["id"].as_str().unwrap(), json!({"status": "interrupted", "finished": stamp().0}))?;
            t["status"] = json!("interrupted");
        }
        let mine = all.iter().find(|t| t["id"] == id.as_str()).ok_or("own ticket vanished")?;
        if status(mine) == "cancelled" {
            say(&format!("[очередь] {id}: отменено, не запускаю"));
            return Err(format!("ticket {id} was cancelled").into());
        }
        if status(mine) == "interrupted" { // this process is alive, so someone judged it dead by mistake
            update(queue, &id, json!({"status": "queued", "finished": null}))?;
            continue;
        }
        let queued: Vec<&Value> = all.iter().filter(|t| status(t) == "queued").collect();
        let ahead = queued.iter().filter(|t| t["id"].as_str().unwrap() < id.as_str()).count();
        let waiting: HashSet<u32> = queued.iter().map(|t| pid(t)).collect();
        let busy = gpu_processes(&ps, &waiting.iter().copied().chain([me]).collect());
        let mut held = !busy.is_empty() || all.iter().any(|t| status(t) == "running");
        if ahead == 0 && busy.is_empty() {
            match try_lock(queue)? {
                Some(lock) => {
                    let started = stamp().0;
                    update(queue, &id, json!({"status": "running", "started": started}))?;
                    say(&format!("[очередь] {id}: запущено {started}"));
                    return Ok(Ticket { queue: queue.to_path_buf(), id, _lock: lock });
                }
                None => held = true,
            }
        }
        let position = 1 + ahead + usize::from(held);
        let why = if busy.is_empty() { "ждёт своей очереди".to_string() } else {
            let names: Vec<String> = busy.iter().map(|p| format!("{} pid {}", p.name, p.pid)).collect();
            format!("ждёт, GPU занят: {}", names.join(", "))
        };
        let message = format!("[очередь] поставлено {id}: позиция {position}, {why}");
        if message != last {
            say(&message);
            update(queue, &id, json!({"position": position, "waiting_for": busy.iter().map(|p| json!({"pid": p.pid, "name": p.name})).collect::<Vec<_>>()}))?;
            last = message;
        }
        std::thread::sleep(poll);
    }
}

impl Ticket {
    pub fn finish(self, code: i32) {
        let finished = stamp().0;
        let fields = json!({"status": if code == 0 { "done" } else { "failed" }, "returncode": code, "finished": finished});
        if let Err(e) = update(&self.queue, &self.id, fields) { eprintln!("[очередь] {}: не записан конец: {e}", self.id); }
        println!("[очередь] {}: завершено {finished}, код {code}", self.id);
    }
}

pub fn cancel(queue: &Path, id: &str) -> Result<()> {
    let t = read(&path_of(queue, id)).ok_or_else(|| format!("нет задания {id}"))?;
    if status(&t) != "queued" { return Err(format!("{id} уже {}: отменить можно только ждущее", status(&t)).into()); }
    update(queue, id, json!({"status": "cancelled", "finished": stamp().0}))?;
    Ok(())
}

fn short(t: &Value) -> String {
    let what = t["config"].as_str().map(String::from).unwrap_or_else(|| {
        t["command"].as_array().map(|c| c.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" ")).unwrap_or_default()
    });
    format!("{}  {what}  (pid {})", t["id"].as_str().unwrap_or("?"), pid(t))
}

pub fn status_text(queue: &Path, procs: &[Proc]) -> String {
    let all = tickets(queue);
    let alive: HashSet<u32> = procs.iter().map(|p| p.pid).collect();
    let live = |t: &&Value| alive.contains(&pid(t));
    let running: Vec<&Value> = all.iter().filter(|t| status(t) == "running").filter(live).collect();
    let queued: Vec<&Value> = all.iter().filter(|t| status(t) == "queued").filter(live).collect();
    let mut lines: Vec<String> = running.iter().map(|t| format!("Идёт: {} с {}", short(t), t["started"].as_str().unwrap_or("?"))).collect();
    let tickets_pids: HashSet<u32> = running.iter().chain(&queued).map(|t| pid(t)).collect();
    let outside = gpu_processes(procs, &tickets_pids);
    for p in &outside { lines.push(format!("Идёт вне очереди: {} pid {}", p.name, p.pid)); }
    let offset = usize::from(!running.is_empty() || !outside.is_empty());
    for (k, t) in queued.iter().enumerate() { lines.push(format!("{}. {}", k + 1 + offset, short(t))); }
    if lines.is_empty() { lines.push("Очередь пуста.".into()); }
    let ended: Vec<&Value> = all.iter().filter(|t| ENDED.contains(&status(t))).collect();
    if !ended.is_empty() {
        lines.push("Последние завершённые:".into());
        for t in &ended[ended.len().saturating_sub(5)..] {
            lines.push(format!("  {} (код {}) {}  {}", status(t), t["returncode"], short(t), t["finished"].as_str().unwrap_or("")));
        }
    }
    lines.join("\n")
}

/// The processes of this machine: names and pids from tasklist; command lines only of Python ones (scripts).
pub fn system_processes() -> Vec<Proc> {
    use std::process::Command;
    #[cfg(windows)]
    fn quiet(c: &mut Command) -> &mut Command { use std::os::windows::process::CommandExt; c.creation_flags(0x0800_0000) }
    #[cfg(not(windows))]
    fn quiet(c: &mut Command) -> &mut Command { c }
    let Ok(out) = quiet(Command::new("tasklist").args(["/fo", "csv", "/nh"])).output() else { return vec![] };
    let mut procs: Vec<Proc> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| {
        let cells: Vec<&str> = l.split("\",\"").map(|c| c.trim_matches('"')).collect();
        Some(Proc { name: cells.first()?.to_string(), pid: cells.get(1)?.parse().ok()?, parent: 0, cmdline: String::new() })
    }).collect();
    // the parent of every process, the command line of the Python ones (scripts)
    let script = "Get-CimInstance Win32_Process | ForEach-Object { $c = if ($_.Name -like 'python*') { $_.CommandLine } else { '' }; \"$($_.ProcessId)`t$($_.ParentProcessId)`t$c\" }";
    if let Ok(out) = quiet(Command::new("powershell").args(["-NoProfile", "-Command", script])).output() {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let mut cells = line.splitn(3, '\t');
            let (Some(pid), Some(parent)) = (cells.next(), cells.next()) else { continue };
            let (Ok(pid), Ok(parent)) = (pid.trim().parse::<u32>(), parent.trim().parse::<u32>()) else { continue };
            let cmd = cells.next().unwrap_or("").trim_end();
            for p in procs.iter_mut().filter(|p| p.pid == pid) {
                p.parent = parent;
                if !cmd.is_empty() { p.cmdline = cmd.to_string(); }
            }
        }
    }
    procs
}

/// The variable a holder of the card (wan search) sets for the runs it starts: they run inside its ticket.
pub const HELD: &str = "WAN_QUEUE_HELD";

/// Whether ticket `id` is running and its process is alive: a child of it may use the card without a ticket.
pub fn held(queue: &Path, id: &str, procs: &[Proc]) -> bool {
    read_existing(&path_of(queue, id)).is_some_and(|t| status(&t) == "running" && procs.iter().any(|p| p.pid == pid(&t)))
}

fn read_existing(path: &Path) -> Option<Value> { if path.exists() { read(path) } else { None } }

/// The gate of main: wait for this run's turn on the real machine, printing as it goes.
pub fn enter(argv: &[String], config: Option<&Path>) -> Result<Ticket> {
    let exe = std::env::current_exe().map(|e| e.display().to_string()).unwrap_or_else(|_| "wan.exe".into());
    let command: Vec<String> = std::iter::once(exe).chain(argv.iter().cloned()).collect();
    let mut say = |m: &str| { use std::io::Write; println!("{m}"); let _ = std::io::stdout().flush(); };
    take_turn(&dir(), std::process::id(), &command, config, &mut system_processes, &mut say, Duration::from_secs(3))
}

/// `wan queue status | cancel <id>`; the exit code.
pub fn main(argv: &[String]) -> i32 {
    match argv.first().map(String::as_str) {
        Some("status") => {
            let me = std::process::id(); // this status command is a wan.exe too
            let procs: Vec<Proc> = system_processes().into_iter().filter(|p| p.pid != me).collect();
            println!("{}", status_text(&dir(), &procs));
            0
        }
        Some("cancel") if argv.len() == 2 => match cancel(&dir(), &argv[1]) {
            Ok(()) => { println!("[очередь] {} отменено", argv[1]); 0 }
            Err(e) => { eprintln!("{e}"); 1 }
        },
        _ => { eprintln!("wan queue status | wan queue cancel <id>\nЛюбой запуск wan с конфигом сам встаёт в очередь."); 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn queue_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wan-queue-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }
    fn wan(pid: u32) -> Proc { Proc { pid, parent: 0, name: "wan.exe".into(), cmdline: "wan.exe --config x.json".into() } }
    fn cmd(s: &str) -> Vec<String> { s.split(' ').map(String::from).collect() }
    fn status_of(queue: &Path, id: &str) -> String {
        tickets(queue).into_iter().find(|t| t["id"] == id).unwrap()["status"].as_str().unwrap().into()
    }
    const FAST: Duration = Duration::from_millis(10);

    #[test]
    fn an_idle_card_runs_at_once_at_position_one() {
        let q = queue_dir("idle");
        let mut said = vec![];
        let t = take_turn(&q, 11, &cmd("wan.exe --config a.json"), None, &mut || vec![wan(11)],
                          &mut |m| said.push(m.to_string()), FAST).unwrap();
        assert_eq!(status_of(&q, &t.id), "running");
        assert!(said.iter().any(|m| m.contains("запущено")), "{said:?}");
        let id = t.id.clone();
        t.finish(0);
        let rec = tickets(&q).into_iter().find(|r| r["id"] == id.as_str()).unwrap();
        assert_eq!((rec["status"].as_str(), rec["returncode"].as_i64()), (Some("done"), Some(0)));
    }

    #[test]
    fn a_second_run_waits_at_position_two_and_starts_after_the_first() {
        let q = queue_dir("second");
        let first_alive = Arc::new(AtomicBool::new(true));
        let a = take_turn(&q, 21, &cmd("wan.exe a"), None, &mut || vec![wan(21)], &mut |_| {}, FAST).unwrap();
        let said = Arc::new(Mutex::new(Vec::<String>::new()));
        let (q2, alive2, said2) = (q.clone(), first_alive.clone(), said.clone());
        let b = std::thread::spawn(move || {
            let mut procs = || if alive2.load(Ordering::SeqCst) { vec![wan(21), wan(22)] } else { vec![wan(22)] };
            let t = take_turn(&q2, 22, &cmd("wan.exe b"), None, &mut procs,
                              &mut |m| said2.lock().unwrap().push(m.to_string()), FAST).unwrap();
            let id = t.id.clone();
            t.finish(3);
            id
        });
        std::thread::sleep(Duration::from_millis(300));
        assert!(said.lock().unwrap().iter().any(|m| m.contains("позиция 2") && m.contains("ждёт")), "{:?}", said.lock().unwrap());
        assert!(!said.lock().unwrap().iter().any(|m| m.contains("запущено")));
        a.finish(0);
        first_alive.store(false, Ordering::SeqCst);
        let b_id = b.join().unwrap();
        assert_eq!(status_of(&q, &b_id), "failed");
    }

    #[test]
    fn a_ticket_of_a_dead_process_is_interrupted_and_blocks_nothing() {
        let q = queue_dir("stale");
        let dead = take_turn(&q, 31, &cmd("wan.exe a"), None, &mut || vec![wan(31)], &mut |_| {}, FAST).unwrap();
        let dead_id = dead.id.clone();
        drop(dead); // the process died: its lock is gone, its ticket still says running
        let t = take_turn(&q, 32, &cmd("wan.exe b"), None, &mut || vec![wan(32)], &mut |_| {}, FAST).unwrap();
        assert_eq!(status_of(&q, &dead_id), "interrupted");
        t.finish(0);
    }

    #[test]
    fn a_gpu_run_outside_the_queue_is_waited_for() {
        let q = queue_dir("outside");
        let mut looks = 0;
        let mut said = vec![];
        let t = take_turn(&q, 41, &cmd("wan.exe a"), None, &mut || {
            looks += 1;
            let sweep = Proc { pid: 4242, parent: 0, name: "wan-ntk-sweep.exe".into(), cmdline: String::new() };
            if looks <= 2 { vec![wan(41), sweep] } else { vec![wan(41)] }
        }, &mut |m| said.push(m.to_string()), FAST).unwrap();
        assert!(said.iter().any(|m| m.contains("wan-ntk-sweep.exe")), "{said:?}");
        t.finish(0);
    }

    #[test]
    fn gpu_processes_are_recognised_by_name_and_command_line() {
        let p = |pid, name: &str, line: &str| Proc { pid, parent: 0, name: name.into(), cmdline: line.into() };
        let procs = vec![
            p(1, "wan.exe", ""), p(2, "wan-ntk-sweep.exe", ""), p(3, "python.exe", "python scripts\\experiments\\run_experiment.py"),
            p(4, "python.exe", "python -m pytest"), p(5, "wan-debug.exe", ""), p(6, "notepad.exe", ""), p(7, "live.exe", ""),
        ];
        let found: Vec<u32> = gpu_processes(&procs, &HashSet::from([5])).iter().map(|p| p.pid).collect();
        assert_eq!(found, vec![1, 2, 3, 7]);
    }

    /// A driver under scripts\experiments\ that starts wan.exe (through the venv's python launcher) is its parent, not
    /// a GPU job beside it: a run never waits for its own ancestors (2026-11-11 a CMA driver held its wan.exe 3 hours).
    #[test]
    fn a_run_does_not_wait_for_the_processes_that_started_it() {
        let p = |pid, parent, name: &str, line: &str| Proc { pid, parent, name: name.into(), cmdline: line.into() };
        let procs = vec![
            p(10, 1, "python.exe", "python scripts\\experiments\\rope_cma.py spec.json"),
            p(11, 10, "python.exe", "python scripts\\experiments\\rope_cma.py spec.json"),
            p(12, 11, "wan.exe", "wan.exe --config g000.json"),
            p(20, 1, "python.exe", "python scripts/experiments/run_experiment.py"),
            p(30, 30, "wan.exe", ""), // a parent loop must not hang the walk
        ];
        let found: Vec<u32> = gpu_processes(&procs, &HashSet::from([12])).iter().map(|p| p.pid).collect();
        assert_eq!(found, vec![20, 30]);
    }

    #[test]
    fn a_cancelled_ticket_stops_its_wait() {
        let q = queue_dir("cancel");
        let a = take_turn(&q, 51, &cmd("wan.exe a"), None, &mut || vec![wan(51)], &mut |_| {}, FAST).unwrap();
        let q2 = q.clone();
        let b = std::thread::spawn(move || {
            take_turn(&q2, 52, &cmd("wan.exe b"), None, &mut || vec![wan(51), wan(52)], &mut |_| {}, FAST).map(|t| t.id.clone())
        });
        std::thread::sleep(Duration::from_millis(200));
        let waiting = tickets(&q).into_iter().find(|t| t["status"] == "queued").unwrap();
        cancel(&q, waiting["id"].as_str().unwrap()).unwrap();
        assert!(b.join().unwrap().is_err());
        a.finish(0);
    }

    #[test]
    fn status_lists_the_running_and_the_waiting_with_positions() {
        let q = queue_dir("status");
        let a = take_turn(&q, 61, &cmd("wan.exe --config a.json"), Some(Path::new("a.json")), &mut || vec![wan(61)], &mut |_| {}, FAST).unwrap();
        let text = status_text(&q, &[wan(61)]);
        assert!(text.contains("Идёт") && text.contains("a.json"), "{text}");
        a.finish(0);
        assert!(status_text(&q, &[]).contains("Очередь пуста"));
    }

    #[test]
    fn a_running_ticket_of_a_live_process_is_held_and_others_are_not() {
        let q = queue_dir("held");
        let t = take_turn(&q, 101, &cmd("wan.exe search"), None, &mut || vec![wan(101)], &mut |_| {}, FAST).unwrap();
        assert!(held(&q, &t.id, &[wan(101), wan(102)]));
        assert!(!held(&q, &t.id, &[wan(102)]), "the holder is dead");
        assert!(!held(&q, "20000101-000000-000000000", &[wan(101)]), "no such ticket");
        let id = t.id.clone();
        t.finish(0);
        assert!(!held(&q, &id, &[wan(101)]), "an ended ticket holds nothing");
    }

    #[test]
    fn a_run_that_starts_after_the_snapshot_is_not_judged_dead() {
        let q = queue_dir("snapshot");
        let (q2, mut first) = (q.clone(), true);
        // the process list is taken while another run is just taking its ticket: that run is not in the list
        let t = take_turn(&q, 71, &cmd("wan.exe a"), None, &mut || {
            if first {
                first = false;
                write(&q2, &json!({"id": "99999999-000000-000000001", "status": "queued", "command": ["wan.exe"], "pid": 72})).unwrap();
            }
            vec![wan(71)]
        }, &mut |_| {}, FAST);
        drop(t);
        assert_eq!(status_of(&q, "99999999-000000-000000001"), "queued");
    }

    #[test]
    fn an_own_ticket_marked_interrupted_while_waiting_is_revived_and_runs() {
        let q = queue_dir("revive");
        let a = take_turn(&q, 81, &cmd("wan.exe a"), None, &mut || vec![wan(81)], &mut |_| {}, FAST).unwrap();
        let q2 = q.clone();
        let first_alive = Arc::new(AtomicBool::new(true));
        let alive2 = first_alive.clone();
        let said = Arc::new(Mutex::new(Vec::<String>::new()));
        let said2 = said.clone();
        let b = std::thread::spawn(move || {
            let mut procs = || if alive2.load(Ordering::SeqCst) { vec![wan(81), wan(82)] } else { vec![wan(82)] };
            take_turn(&q2, 82, &cmd("wan.exe b"), None, &mut procs,
                      &mut |m| said2.lock().unwrap().push(m.to_string()), FAST).map(|t| { let id = t.id.clone(); t.finish(0); id })
        });
        std::thread::sleep(Duration::from_millis(200));
        let mine = tickets(&q).into_iter().find(|t| t["pid"] == 82).unwrap();
        update(&q, mine["id"].as_str().unwrap(), json!({"status": "interrupted"})).unwrap(); // judged dead by mistake
        std::thread::sleep(Duration::from_millis(100));
        a.finish(0);
        first_alive.store(false, Ordering::SeqCst);
        let id = b.join().unwrap().expect("the run must start");
        assert_eq!(status_of(&q, &id), "done");
        assert!(!said.lock().unwrap().iter().any(|m| m.contains("wan.exe pid 82")), "{:?}", said.lock().unwrap());
    }
}

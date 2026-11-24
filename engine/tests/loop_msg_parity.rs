//! MessageController + message gather parity vs goldens/message_controller.npz.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use grouille_inkling::loop_msg::{
    gather_messages, message_controller, MessageControllerWeights, GATE_RANK,
};
use grouille_inkling::loop_sched::unpack_message_queries;
use grouille_inkling::ops::HIDDEN;
use safetensors::tensor::SafeTensors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn ckpt() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("model-00001-of-00001.safetensors")
}

fn load_f32(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let mut npz = npyz::npz::NpzArchive::new(f)?;
    let arr = npz
        .by_name(key)?
        .with_context(|| format!("missing {key}"))?;
    Ok(arr.into_vec()?)
}

fn f32s(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name)?;
    Ok(t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn message_controller_and_gather() -> Result<()> {
    let path = repo_root().join("goldens/message_controller.npz");
    if !path.exists() {
        bail!(
            "missing {}; run spike/dump_message_controller.py",
            path.display()
        );
    }
    let x = load_f32(&path, "x")?;
    let message = load_f32(&path, "message")?;
    let out_ref = load_f32(&path, "out")?;
    let down = load_f32(&path, "down")?;
    let forget = load_f32(&path, "forget")?;
    let gain = load_f32(&path, "gain")?;
    assert_eq!(x.len(), HIDDEN);
    assert_eq!(down.len(), GATE_RANK * 2 * HIDDEN);

    let w = MessageControllerWeights {
        down: &down,
        forget: &forget,
        gain: &gain,
    };
    let mut out = vec![0f32; HIDDEN];
    message_controller(&x, &message, &w, &mut out);
    let mut maxabs = 0f32;
    for (a, b) in out.iter().zip(out_ref.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    eprintln!("message_controller maxabs={maxabs}");
    assert!(maxabs < 0.05, "controller maxabs={maxabs}");

    let bank = load_f32(&path, "bank")?;
    let raw = load_f32(&path, "raw_states")?;
    let queries = load_f32(&path, "queries")?;
    let messages_ref = load_f32(&path, "messages")?;
    let n_states = bank.len() / HIDDEN;
    assert_eq!(raw.len(), bank.len());
    let mut bank_pred = vec![0f32; bank.len()];
    for i in 0..n_states {
        grouille_inkling::ops::rms_norm(
            &raw[i * HIDDEN..(i + 1) * HIDDEN],
            None,
            &mut bank_pred[i * HIDDEN..(i + 1) * HIDDEN],
        );
    }
    let mut bmax = 0f32;
    for (a, b) in bank_pred.iter().zip(bank.iter()) {
        bmax = bmax.max((a - b).abs());
    }
    eprintln!("bank rms_norm maxabs={bmax}");
    assert!(bmax < 1e-4, "bank rms maxabs={bmax}");

    let mut messages = vec![0f32; 4 * HIDDEN];
    gather_messages(&bank, n_states, &queries, &mut messages);
    let mut gmax = 0f32;
    for (a, b) in messages.iter().zip(messages_ref.iter()) {
        gmax = gmax.max((a - b).abs());
    }
    eprintln!("gather_messages maxabs={gmax}");
    assert!(gmax < 1e-4, "gather maxabs={gmax}");

    let file = std::fs::File::open(ckpt())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let cross = f32s(&st, "model.llm.middle_cycle_cross_pass_queries")?;
    let kv = f32s(&st, "model.llm.middle_cycle_cross_pass_kv_queries")?;
    let all = unpack_message_queries(&cross, &kv);
    let q0 = &all[..4 * HIDDEN];
    let mut qmax = 0f32;
    for (a, b) in q0.iter().zip(queries.iter()) {
        qmax = qmax.max((a - b).abs());
    }
    eprintln!("unpack message_queries[0] maxabs={qmax}");
    assert!(qmax < 1e-6, "unpack maxabs={qmax}");
    Ok(())
}

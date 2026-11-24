//! HuggingFace `tokenizer.json` wrapper for Inkling 2 Mini.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;


pub struct InklingTokenizer {
    inner: Tokenizer,
}

impl InklingTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    /// Encode text to model token ids (`i32` for embed lookups).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Decode token ids to text.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let uids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&uids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

/// Default HF snapshot path for Inkling 2 Mini on this machine.
pub fn default_tokenizer_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini")
        .join("snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7")
        .join("tokenizer.json")
}

/// Load the default snapshot tokenizer if present.
pub fn load_default_tokenizer() -> Result<InklingTokenizer> {
    let path = default_tokenizer_path();
    InklingTokenizer::from_file(&path).with_context(|| format!("tokenizer at {}", path.display()))
}

/// Minimal chat-template fragment for a single user turn (no tools).
///
/// Matches HF/MLX system+user framing (no BOS). At `thinking_effort == 0` we
/// open `<|message_model|><|content_text|>` so greedy decode skips the scratchpad
/// and answers immediately; otherwise leave the model turn unopened (MLX default)
/// so the model emits `<|message_model|>` then thinking/text itself.
pub fn format_user_prompt(user_text: &str, thinking_effort: f32) -> String {
    format_chat_turns(&[(ChatRole::User, user_text)], thinking_effort, None)
}

/// User turn to append onto a transcript that already ends at the previous
/// assistant `<|end_message|>` (REPL KV continue). Avoids re-encoding history.
pub fn format_user_turn_delta(user_text: &str, thinking_effort: f32) -> String {
    let open = if thinking_effort <= 0.0 {
        "<|message_model|><|content_text|>"
    } else {
        ""
    };
    format!("<|message_user|><|content_text|>{user_text}<|end_message|>{open}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

/// Multi-turn chat wrap: prior user/assistant pairs, then the latest user turn.
///
/// Assistant turns are stored as plain answer text and re-wrapped as
/// `<|message_model|><|content_text|>…<|end_message|>`.
/// `system_extra`, when set, is inserted as an additional system message after
/// the effort/style headers.
pub fn format_chat_turns(
    turns: &[(ChatRole, &str)],
    thinking_effort: f32,
    system_extra: Option<&str>,
) -> String {
    assert!(!turns.is_empty(), "need at least one turn");
    let effort = if thinking_effort <= 0.0 {
        "0".to_string()
    } else {
        format!("{thinking_effort}")
    };
    let open = if thinking_effort <= 0.0 {
        "<|message_model|><|content_text|>"
    } else {
        ""
    };
    let style = if thinking_effort <= 0.0 {
        "<|message_system|><|content_text|>Answer briefly and directly.<|end_message|>"
    } else {
        ""
    };
    let mut out = format!(
        "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>{style}"
    );
    if let Some(sys) = system_extra {
        if !sys.is_empty() {
            out.push_str("<|message_system|><|content_text|>");
            out.push_str(sys);
            out.push_str("<|end_message|>");
        }
    }
    for &(role, text) in turns {
        match role {
            ChatRole::User => {
                out.push_str("<|message_user|><|content_text|>");
                out.push_str(text);
                out.push_str("<|end_message|>");
            }
            ChatRole::Assistant => {
                out.push_str("<|message_model|><|content_text|>");
                out.push_str(text);
                out.push_str("<|end_message|>");
            }
        }
    }
    // Only open the next model turn after a user message.
    if matches!(turns.last().map(|t| t.0), Some(ChatRole::User)) {
        out.push_str(open);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_turn_delta_opens_text_at_effort_zero() {
        let d = format_user_turn_delta("hi", 0.0);
        assert!(d.starts_with("<|message_user|><|content_text|>hi<|end_message|>"));
        assert!(d.ends_with("<|message_model|><|content_text|>"));
        let think = format_user_turn_delta("hi", 1.0);
        assert!(think.ends_with("<|end_message|>"));
        assert!(!think.contains("<|message_model|>"));
    }
}

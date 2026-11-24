//! Inkling 2 Mini on Metal: the GPU engine, the host reference it is checked
//! against, and the drafter.

pub mod attn;
pub mod cache;
pub mod decode_visit;
pub mod decoder;
pub mod drafter;
pub mod flags;
pub mod gpu;
pub mod gpk;
pub mod head;
pub mod layer0;
pub mod layer5;
pub mod loop_msg;
pub mod loop_sched;
pub mod visit;
pub mod weights;
pub mod metal_draft;
pub mod metal_fuse;
pub mod metal_glue;
pub mod metal_q8;
pub mod metal_fuse_full;
pub mod metal_moe;
pub mod metal_mxfp8;
pub mod metal_nvfp4;
pub mod metal_ops;
pub mod metal_prompt;
pub mod metal_resident;
pub mod moe;
pub mod mxfp8;
pub mod nvfp4;
pub mod ops;
pub mod prefill;
pub mod route;
pub mod speculate;
pub mod tokenize;

pub use cache::{DecodeCaches, VisitCache};
pub use loop_sched::{LAYER_INDICES, N_VISITS, CORE_END, CORE_START, NUM_PASSES};
pub use head::{
    CONTENT_TEXT_TOKEN_ID, CONTENT_THINKING_TOKEN_ID, END_MESSAGE_TOKEN_ID, EOS_TOKEN_ID,
    MESSAGE_MODEL_TOKEN_ID, PAD_TOKEN_ID, UNPADDED_VOCAB, VOCAB,
};
pub use tokenize::{
    default_tokenizer_path, format_chat_turns, format_user_prompt, format_user_turn_delta,
    load_default_tokenizer, ChatRole, InklingTokenizer,
};
pub use layer0::layer0_prefill;
pub use metal_mxfp8::Mxfp8Gemv;
pub use metal_nvfp4::Nvfp4Gemv;
pub use moe::{sparse_moe_dispatch, sparse_moe_dispatch_into, sparse_moe_layer0};
pub use mxfp8::{dequant_mxfp8_rows, mxfp8_gemv};
pub use nvfp4::{dequant_nvfp4_rows, nvfp4_gemv, NVFP4_MAX};
pub use decode_visit::{
    decode_logits_metal, decode_logits_metal_into, decode_next_id_metal, extend_prompt_metal,
    decode_sequence_logits_metal, generate_argmax_metal,
    generate_argmax_metal_stop, generate_sample_metal, generate_sample_metal_top_k,
    run_decode_visits, run_decode_visits_cached, run_decode_visits_metal,
    apply_presence_penalty, recycle_decode_caches, recycle_decode_caches_with_logits,
    run_decode_visits_metal_cached, sample_logits, sample_logits_top_k, sample_logits_top_k_p,
    commit_block_metal, verify_block_metal, warm_caches_pooled, warm_prompt_metal,
    warm_prompt_taps_metal, BlockPass, DecodeTokenState, WarmPrompt, XorShift32,
};
pub use prefill::{finalize_hidden, prefill_hidden_metal, prefill_logits_metal, HeadBufs};
pub use visit::{
    run_visits, run_visits_cached, run_visits_into_caches, run_visits_into_caches_metal,
    run_visits_metal, run_visits_metal_cached, MetalAccel, VisitState, WeightCache,
};

#[cfg(test)]
mod long_attn_tests;

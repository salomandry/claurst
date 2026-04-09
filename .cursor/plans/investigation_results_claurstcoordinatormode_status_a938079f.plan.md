---
name: "Investigation Results: CLAURSTCOORDINATORMODE Status"
overview: Investigate CLAURST_COORDINATOR_MODE implementation status and identify gaps
todos: []
isProject: false
---

# Investigation Results: CLAURST_COORDINATOR_MODE Status

## Summary
The `CLAURST_COORDINATOR_MODE` feature is **partially implemented** but **not functional end-to-end**. The infrastructure exists in Rust, but it's not integrated into the main code flow.

## What Exists (Rust Implementation)

### 1. Coordinator Module (`src-rust/crates/query/src/coordinator.rs`)
- `COORDINATOR_ENV_VAR` constant defined
- `is_coordinator_mode()` function to check env var
- `AgentMode` enum (Coordinator, Worker, Normal)
- `filter_tools_for_mode()` for tool filtering
- `coordinator_system_prompt()` with coordinator instructions
- `coordinator_user_context()` for worker context
- `match_session_mode()` and `match_session_mode_from_agent_mode()` for session resume alignment
- Comprehensive unit tests

### 2. System Prompt Integration (`src-rust/crates/core/src/system_prompt.rs`)
- `SystemPromptOptions` has `coordinator_mode: bool` field
- `COORDINATOR_SYSTEM_PROMPT` constant with coordinator instructions
- Conditional injection when `opts.coordinator_mode` is true

## What's Missing (Integration Gaps)

### 1. CLI Not Reading Environment Variable
The CLI doesn't check `CLAURST_COORDINATOR_MODE` env var and set `coordinator_mode` in `SystemPromptOptions`.

### 2. Query Config Not Using Coordinator Mode
In `src-rust/crates/query/src/lib.rs`, the `build_system_prompt()` function always uses default `coordinator_mode: false`.

### 3. Session Storage Doesn't Track Coordinator Mode
`ConversationSession` struct has no field to persist coordinator mode across sessions.

### 4. No TypeScript Implementation
The spec references `src/coordinator/coordinatorMode.ts` which doesn't exist in this clean-room Rust implementation.

## Required Changes to Make It Work

1. **CLI Integration**: Read `CLAURST_COORDINATOR_MODE` env var and pass to query config
2. **Query Config**: Add coordinator_mode field and use it in system prompt building
3. **Session Persistence**: Add coordinator_mode field to ConversationSession
4. **Session Resume**: Call `match_session_mode()` when resuming sessions

## Recommendation
The coordinator mode infrastructure is well-designed but not wired up. It would require approximately 2-3 files modified to complete the integration.
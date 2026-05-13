# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? | Made By |
|---|------|-------|----------|--------|-----------|------------|---------|
| D001 | M001 | arch | Systolic array dataflow style | Output-stationary (each PE accumulates one output element, weights stream through) | AWN GEMM shapes have K (192-320) large relative to M/N (64-128); output-stationary naturally hides K-dimension accumulation latency; same accumulation order as behavioral model guarantees zero rounding divergence for bit-exact verification | No | collaborative |
| D002 | M001 | arch | Memory architecture for systolic array feeders | Dedicated dual-port BRAMs for row (A-matrix) and column (B-matrix) feeders | Systolic array needs simultaneous row and column reads every cycle; existing single-port global_buffer.v would halve throughput | No | collaborative |
| D003 | M001 | arch | im2col implementation placement | Hardware im2col FSM in PL fabric (not software on PS ARM) | Software im2col adds 350-700 us overhead — nearly the entire 500 us WiFi latency budget; hardware im2col adds ~200 LUTs and eliminates this overhead | No | collaborative |
| D004 | M001 | pattern | Verification strategy for systolic array | Three-tier: (1) per-PE unit tests, (2) randomized matrix tests, (3) end-to-end 38-op hex pipeline | Existing hex-file pattern is proven across 9 modules but only covers AWN-specific inputs; per-PE and randomized tests catch corner cases in accumulation, overflow, and tiling boundaries per user requirement | No | collaborative |

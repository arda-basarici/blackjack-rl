# Overnight DQN ablation sweep — runs many combinations back-to-back, unattended.
# Each run auto-saves a record to runs/ and tee's its log to logs/, so nothing is lost and the
# morning analysis (notebook + records) can compare them all.
#
# Usage (from blackjack-rl/, with the Phase-3 venv active):
#     .\.venv\Scripts\Activate.ps1
#     .\scripts\overnight_sweep.ps1
#
# Notes:
# - 1,000,000 episodes per run: agreement plateaus by ~500k (flat 300k->2M in every run so far),
#   so 1M is safely converged and ~halves each run, letting all combinations fit overnight.
# - Everything not varied is held fixed: linear eps 0.3->0, 100k-hand eval, seed 42, progress 10k.
# - Args override left-to-right, so appending e.g. "--seed 43" or "--lr 1e-4" beats the common default.
# - Comment out any $variants line to skip it. ~22 runs; rough total 5-7h (the 256x256 ones are slow).

$ErrorActionPreference = "Continue"   # one failed run must not kill the batch

$common = @(
    "--episodes", "1000000",
    "--epsilon-schedule", "linear", "--epsilon-start", "0.3", "--epsilon-end", "0.0",
    "--eval-hands", "100000", "--seed", "42", "--progress-every", "10000"
)

# label                          extra args
$variants = @(
    # --- core ablation: one factor at a time from baseline (scalar, vanilla, 64x64) ---
    @("scalar vanilla 64 (baseline)", @("--encoding", "scalar")),
    @("scalar + double-dqn",          @("--encoding", "scalar", "--double-dqn")),
    @("onehot vanilla 64",            @("--encoding", "onehot")),
    @("onehot + double-dqn",          @("--encoding", "onehot", "--double-dqn")),

    # --- capacity axis (net size) ---
    @("scalar 32x32",                 @("--encoding", "scalar", "--hidden", "32,32")),
    @("scalar 128x128",               @("--encoding", "scalar", "--hidden", "128,128")),
    @("scalar 256x256",               @("--encoding", "scalar", "--hidden", "256,256")),
    @("scalar 64x64x64 (deeper)",     @("--encoding", "scalar", "--hidden", "64,64,64")),
    @("onehot 256x256",               @("--encoding", "onehot", "--hidden", "256,256")),

    # --- learning rate (tests the wobble/instability) ---
    @("scalar lr 3e-4",               @("--encoding", "scalar", "--lr", "0.0003")),
    @("scalar lr 1e-4",               @("--encoding", "scalar", "--lr", "0.0001")),
    @("onehot lr 1e-4",               @("--encoding", "onehot", "--lr", "0.0001")),

    # --- replay ratio / target cadence (other stabilizers) ---
    @("scalar train_every 1",         @("--encoding", "scalar", "--train-every", "1")),
    @("scalar target-sync 250",       @("--encoding", "scalar", "--target-sync", "250")),

    # --- multiple seeds (quantify run-to-run wobble for honest ranking) ---
    @("scalar seed 43",               @("--encoding", "scalar", "--seed", "43")),
    @("scalar seed 44",               @("--encoding", "scalar", "--seed", "44")),
    @("onehot seed 43",               @("--encoding", "onehot", "--seed", "43")),
    @("onehot seed 44",               @("--encoding", "onehot", "--seed", "44")),

    # --- exploring-starts capstone (forced coverage) + extensions ---
    @("ES scalar",                    @("--encoding", "scalar", "--exploring-starts")),
    @("ES onehot",                    @("--encoding", "onehot", "--exploring-starts")),
    @("ES onehot + lr 1e-4",          @("--encoding", "onehot", "--exploring-starts", "--lr", "0.0001")),
    @("ES onehot + double-dqn",       @("--encoding", "onehot", "--exploring-starts", "--double-dqn"))
)

$total = $variants.Count
$start = Get-Date
for ($i = 0; $i -lt $total; $i++) {
    $label   = $variants[$i][0]
    $runArgs = $common + $variants[$i][1]
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "[$($i + 1)/$total]  $label   (started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
    Write-Host "  args: $($variants[$i][1] -join ' ')"
    Write-Host "================================================================"
    python -m blackjack_rl.dqn.experiment @runArgs
}
Write-Host ""
Write-Host "ALL $total RUNS DONE  (elapsed $((Get-Date) - $start))"

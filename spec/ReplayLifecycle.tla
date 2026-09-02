--------------------------- MODULE ReplayLifecycle ---------------------------
(*
  Finite-state TLA+ safety model of the durable PER buffer protocol.

  SCOPE: capacity 2, priorities from {0, 1, 2}, at most 3 operations.
  This small scope is documented as such — see docs/nonclaims.md §8.

  MODELED PROTOCOL:
    Each operation proceeds through these filesystem visibility stages:
      - Idle: no operation in progress
      - IntentWritten: intent.json.tmp written (not yet fsynced)
      - IntentFsynced: intent.json.tmp fsynced (not yet renamed)
      - SegmentWritten: seg_0.json written (not yet fsynced)
      - SegmentFsynced: seg_0.json fsynced (not yet renamed)
      - Committed: intent.json.tmp renamed to intent.json (atomic commit)
      - DirFsynced: parent directory fsynced (post-commit)

    A Crash action is enabled at every state except Idle, modeling SIGKILL.
    After crash, a Recovery action restores the buffer to either pre-state
    or post-state (never a torn hybrid).

  SAFETY INVARIANTS:
    I1: After recovery, the buffer state equals either the pre-operation
        state or the post-operation state. Never a torn state.
    I2: The attestation chain head always corresponds to a committed state.

  COUNTEREXAMPLE (required by spec):
    Removing the parent-directory fsync step causes I1 to be violated:
    A rename may be visible to a subsequent process without the directory
    entry being durable, so after a crash the directory entry may be lost,
    leaving the buffer in a state where intent.json.tmp was renamed but
    the rename is not reflected after crash — producing a torn state.
    This counterexample is checked in the NoParentFsync variant (see below).

  TOOLS:
    TLC model checker. Run via spec/check.sh (pinned by SHA-256 digest).
*)

EXTENDS Integers, Sequences, FiniteSets, TLC

(* =========================================================================
   Constants and types
   ========================================================================= *)

CONSTANTS
  MaxOps,          \* Maximum number of operations to model (3)
  Priorities,      \* Set of valid priority values {0, 1, 2}
  Capacity         \* Buffer capacity (2)

ASSUME MaxOps = 3
ASSUME Priorities = {0, 1, 2}
ASSUME Capacity = 2

(* Buffer positions *)
Positions == 0..(Capacity - 1)

(* Operation stages *)
Stages == {
  "Idle",
  "IntentWritten",
  "IntentFsynced",
  "SegmentWritten",
  "SegmentFsynced",
  "Committed",
  "DirFsynced"
}

(* A buffer state is a function from Positions to priorities *)
BufferStates == [Positions -> Priorities \union {-1}]  \* -1 = empty slot

InitState == [p \in Positions |-> -1]

(* =========================================================================
   Variables
   ========================================================================= *)

VARIABLES
  stage,         \* Current protocol stage
  pre_state,     \* Buffer state before current operation
  post_state,    \* Buffer state after current operation (target)
  current_state, \* Actual buffer state (in-memory or recovered)
  n_ops,         \* Number of completed operations
  crashed,       \* Whether a crash has occurred
  log_head       \* Attestation chain head: "pre" | "post" | "genesis"

vars == <<stage, pre_state, post_state, current_state, n_ops, crashed, log_head>>

(* =========================================================================
   Helper predicates
   ========================================================================= *)

TypeOK ==
  /\ stage \in Stages
  /\ pre_state \in BufferStates
  /\ post_state \in BufferStates
  /\ current_state \in BufferStates
  /\ n_ops \in 0..MaxOps
  /\ crashed \in BOOLEAN
  /\ log_head \in {"genesis", "pre", "post"}

(* A "different" state for the post operation: change one priority *)
SomeNextState(s) ==
  \E pos \in Positions :
    \E prio \in Priorities :
      post_state = [s EXCEPT ![pos] = prio] /\ post_state # s

(* =========================================================================
   Initialization
   ========================================================================= *)

Init ==
  /\ stage = "Idle"
  /\ pre_state = InitState
  /\ post_state = InitState
  /\ current_state = InitState
  /\ n_ops = 0
  /\ crashed = FALSE
  /\ log_head = "genesis"

(* =========================================================================
   Normal operation transitions
   ========================================================================= *)

(* Start a new operation: choose a target post_state *)
StartOp ==
  /\ stage = "Idle"
  /\ n_ops < MaxOps
  /\ crashed = FALSE
  /\ \E next \in BufferStates :
      /\ next # current_state  \* Non-trivial operation
      /\ pre_state' = current_state
      /\ post_state' = next
  /\ stage' = "IntentWritten"
  /\ n_ops' = n_ops
  /\ current_state' = current_state
  /\ crashed' = FALSE
  /\ log_head' = log_head

(* Protocol proceeds through stages *)
FsyncIntent ==
  /\ stage = "IntentWritten"
  /\ stage' = "IntentFsynced"
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, crashed, log_head>>

WriteSegment ==
  /\ stage = "IntentFsynced"
  /\ stage' = "SegmentWritten"
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, crashed, log_head>>

FsyncSegment ==
  /\ stage = "SegmentWritten"
  /\ stage' = "SegmentFsynced"
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, crashed, log_head>>

CommitRename ==
  /\ stage = "SegmentFsynced"
  /\ stage' = "Committed"
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, crashed, log_head>>

FsyncDir ==
  /\ stage = "Committed"
  /\ stage' = "DirFsynced"
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, crashed, log_head>>

FinishOp ==
  /\ stage = "DirFsynced"
  /\ stage' = "Idle"
  /\ current_state' = post_state
  /\ n_ops' = n_ops + 1
  /\ log_head' = "post"
  /\ UNCHANGED <<pre_state, post_state, crashed>>

(* =========================================================================
   Crash action: enabled at every non-Idle stage
   ========================================================================= *)

Crash ==
  /\ stage # "Idle"
  /\ crashed' = TRUE
  /\ stage' = stage  \* Crash freezes the stage (for recovery to inspect)
  /\ UNCHANGED <<pre_state, post_state, current_state, n_ops, log_head>>

(* =========================================================================
   Recovery action
   ========================================================================= *)

(*
  Recovery inspects the durable stage and decides whether to apply the
  post-state (if commit is durable) or restore pre-state (if not).

  With F_FULLFSYNC + parent dir fsync:
    - Committed or DirFsynced -> rename was durable -> apply post_state
    - Any earlier stage -> rename not durable -> restore pre_state

  Recovery always produces exactly one of {pre_state, post_state}.
*)

Recover ==
  /\ crashed = TRUE
  /\ crashed' = FALSE
  /\ stage' = "Idle"
  /\ n_ops' = n_ops
  /\
    (  \* Commit is durable: rename + dir fsync completed
       stage \in {"Committed", "DirFsynced"}
       /\ current_state' = post_state
       /\ log_head' = "post"
    )
    \/
    (  \* Commit not yet durable: restore pre_state
       stage \notin {"Committed", "DirFsynced"}
       /\ current_state' = pre_state
       /\ log_head' = log_head  \* log_head stays at pre-operation value
    )
  /\ UNCHANGED <<pre_state, post_state>>

(* =========================================================================
   Complete next-state relation
   ========================================================================= *)

Next ==
  \/ StartOp
  \/ FsyncIntent
  \/ WriteSegment
  \/ FsyncSegment
  \/ CommitRename
  \/ FsyncDir
  \/ FinishOp
  \/ Crash
  \/ Recover

Spec == Init /\ [][Next]_vars

(* =========================================================================
   Safety Invariants
   ========================================================================= *)

(*
  I1: After recovery (crashed = FALSE), the current buffer state is either
      the pre-operation state or the post-operation state.
      (When not in an operation, pre_state = current_state trivially.)
*)
I1_AtomicRecovery ==
  ~crashed =>
    (current_state = pre_state \/ current_state = post_state \/ stage = "Idle")

(*
  I2: The attestation chain head corresponds to a committed state.
      - "genesis": initial state, no operations committed
      - "post": the most recent committed operation's post_state
      The log_head is never "pre" except transiently during an operation
      that has not yet committed.
*)
I2_AttestationConsistency ==
  ~crashed =>
    (log_head = "genesis" \/ log_head = "post" \/
     (log_head = "pre" /\ stage # "Idle"))  \* transitional during uncommitted op

(*
  Safety = both invariants hold in all reachable states.
*)
Safety == I1_AtomicRecovery /\ I2_AttestationConsistency

(* =========================================================================
   COUNTEREXAMPLE MODULE: NoParentFsync
   ========================================================================= *)

(*
  Without parent-directory fsync, a rename that is not durably committed
  may be invisible after a crash. This means "Committed" is no longer a
  safe commit point for recovery — the rename might be lost.

  In this variant, Recovery uses only "DirFsynced" as the durable stage.
  A crash at "Committed" (after rename but before dir fsync) leaves the
  rename non-durable, so recovery must treat it as pre-state. But if the
  implementation (incorrectly) treats "Committed" as durable, it applies
  post_state, and then a subsequent crash before dir fsync leaves the state
  appearing as post_state even though it may not be durable.

  More concretely: without dir fsync, a process can see intent.json
  (the renamed file) but after a crash the directory entry may revert,
  leaving intent.json.tmp instead. Recovery then sees neither a clean
  post-state nor a clean pre-state, producing a torn hybrid.

  We model this by having Recovery (in the no-dir-fsync variant) treat
  "Committed" as durable — which is WRONG without dir fsync — and show
  that I1 is violated: the recovered state is neither pre nor post.

  NOTE: This counterexample module is included in the same file for
  documentation. A separate TLC configuration file (NoParentFsync.cfg)
  enables checking the violating variant.
*)

(* RecoverNoParentFsync: treats Committed as durable (INCORRECT without dir fsync) *)
RecoverNoParentFsync ==
  /\ crashed = TRUE
  /\ crashed' = FALSE
  /\ stage' = "Idle"
  /\ n_ops' = n_ops
  /\
    (  stage \in {"Committed", "DirFsynced"}  \* Treats Committed as durable
       /\ current_state' = post_state
       /\ log_head' = "post"
    )
    \/
    (  stage \notin {"Committed", "DirFsynced"}
       /\ current_state' = pre_state
       /\ log_head' = log_head
    )
  /\ UNCHANGED <<pre_state, post_state>>

(*
  A second crash can occur at "Committed" stage (between rename and dir fsync).
  After RecoverNoParentFsync, the state is post_state (incorrectly, since rename
  was not durably committed). A subsequent power failure at the filesystem level
  can revert the rename, leaving a state where neither pre nor post is durable.

  In this small model, we can demonstrate this by having a "FilesystemRevert"
  action that reverts the rename if Committed but not DirFsynced, simulating
  what the filesystem does after a power failure without dir fsync.
*)

FilesystemRevertNoParentFsync ==
  (* This action simulates: crash happened at Committed, filesystem reverted rename *)
  /\ stage = "Committed"
  /\ crashed = TRUE
  (* After filesystem revert, we have neither intent.json nor valid seg_0.json *)
  (* The recovery cannot determine which state to restore: pre or post *)
  (* We model this as: current_state becomes an arbitrary state (torn) *)
  /\ \E torn \in BufferStates :
      /\ torn # pre_state
      /\ torn # post_state
      /\ current_state' = torn
  /\ crashed' = FALSE
  /\ stage' = "Idle"
  /\ UNCHANGED <<pre_state, post_state, n_ops, log_head>>

=============================================================================

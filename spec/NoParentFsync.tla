---------------------------- MODULE NoParentFsync ----------------------------
(*
  The counterexample variant of ReplayLifecycle: recovery that treats the
  renamed intent file as durable before the parent directory has been
  fsynced. After a crash at "Committed" the filesystem may revert the
  rename, and a recovery that already applied the post-state has produced
  a state that is neither pre nor post. TLC must report a violation of
  I1_AtomicRecovery for this module; the correct protocol in
  ReplayLifecycle.tla must not. spec/check.sh --with-counterexample runs
  both.
*)
EXTENDS ReplayLifecycle

NextNoParentFsync ==
  \/ StartOp
  \/ FsyncIntent
  \/ WriteSegment
  \/ FsyncSegment
  \/ CommitRename
  \/ FsyncDir
  \/ FinishOp
  \/ Crash
  \/ RecoverNoParentFsync
  \/ FilesystemRevertNoParentFsync

SpecNoParentFsync == Init /\ [][NextNoParentFsync]_vars

=============================================================================

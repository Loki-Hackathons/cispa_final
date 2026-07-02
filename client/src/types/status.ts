export interface GpuSummary {
  used: number;
  team_jobs: number;
}

export interface NodePack {
  label: string;
  job_ids: string[];
  gpu_total: number;
  gpus_per_node: number;
}

export interface GpuSchedule {
  gpus_per_node: number;
  running_gpus: number;
  pending_gpus: number;
  pending_packs: NodePack[];
}

export interface JobProgress {
  step: number | null;
  total_steps: number | null;
  unit: string | null;
  phase: string | null;
  message: string | null;
  progress_pct: number | null;
  heartbeat_stale: boolean;
  task_id: string | null;
  attempt: number | null;
  status: string | null;
}

export interface CommandChip {
  label: string;
  command: string;
}

export interface SlurmJob {
  job_id: string;
  user: string;
  name: string;
  gpus: number | null;
  state: string;
  elapsed_seconds: number | null;
  time_limit_seconds: number | null;
  eta_seconds: number | null;
  eta_kind: string | null;
  partition: string | null;
  queue_position: number | null;
  priority: number | null;
  reason: string | null;
  start_time: string | null;
  compatible_with: string[];
  progress: JobProgress | null;
  commands: CommandChip[];
}

export interface TaskStatus {
  task_id: string;
  owner: string | null;
  last_score: number | null;
  score_delta: number | null;
  score_history: number[];
  attempt: number | null;
  submit_cooldown_seconds: number;
  query_cooldown_seconds: number;
  submit_ready: boolean;
  query_ready: boolean;
  updated_at: string | null;
  wandb_url: string | null;
  commands: CommandChip[];
}

export interface NextAction {
  kind: string;
  priority: number;
  message: string;
  owner: string | null;
  task_id: string | null;
  job_id: string | null;
}

export interface FailedJob {
  job_id: string;
  user: string;
  name: string;
  state: string;
  exit_code: string | null;
  ended_at: string | null;
  log_err: string | null;
  commands: CommandChip[];
}

export interface OwnerSummary {
  owner: string;
  running_jobs: number;
  pending_jobs: number;
  task_ids: string[];
  submit_ready_tasks: string[];
  query_ready_tasks: string[];
}

export interface ClusterStatus {
  partition: string;
  nodes_alloc: number | null;
  nodes_idle: number | null;
  nodes_total: number | null;
  gpus_alloc: number | null;
  gpus_idle: number | null;
  gpus_total: number | null;
  team_pending: number;
  note: string | null;
}

export interface LeaderboardRow {
  task_id: string;
  team_rank: number | null;
  team_score: number | null;
  leader_score: number | null;
  gap: number | null;
  updated_at: string | null;
}

export interface DashboardStatus {
  mode: string;
  timestamp: string;
  refresh_seconds: number;
  account: string;
  gpu_summary: GpuSummary;
  gpu_schedule: GpuSchedule;
  slurm_jobs: SlurmJob[];
  tasks: TaskStatus[];
  next_actions: NextAction[];
  failed_jobs: FailedJob[];
  owners: OwnerSummary[];
  cluster: ClusterStatus | null;
  leaderboard: LeaderboardRow[];
  leaderboard_page_url?: string;
  warnings: string[];
}

export interface HealthResponse {
  ok: boolean;
  mode: string;
}

export interface HistoryEvent {
  ts: string;
  kind: string;
  task_id: string | null;
  owner: string | null;
  status?: string;
  score?: number;
  file?: string;
  note?: string;
  method?: string;
  extra?: Record<string, unknown>;
}

export interface Task1AttemptSummary {
  id: string;
  created_at: string | null;
  method: string | null;
  note: string | null;
  "tpr_at_0.1pct_fpr": number | null;
  n_documents: number | null;
  n_tokens: number | null;
}

export interface Task1Document {
  document_id: string;
  token_pieces: string[];
  labels: number[];
  scores: number[];
}

export interface Task1Bundle extends Task1AttemptSummary {
  documents: Task1Document[];
}

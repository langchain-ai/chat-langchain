type Optional<T> = T | null | undefined;
export type Metadata = Optional<Record<string, unknown>>;
export interface Thread {
    thread_id: string;
    created_at: string;
    updated_at: string;
    metadata: Metadata;
}

export type Source = {
  url: string;
  title: string;
};

export type Message = {
  id: string;
  createdAt?: Date;
  content: string;
  type: "system" | "human" | "ai" | "function";
  runId?: string;
  sources?: Source[];
  name?: string;
  function_call?: { name: string };
};
export type Feedback = {
  feedback_id: string;
  run_id: string;
  key: string;
  score: number;
  comment?: string;
};
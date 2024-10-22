export type Source = {
  url: string;
  title: string;
};

export type Message = {
  id: string;
  createdAt?: Date;
  content: string;
  type: "system" | "human" | "ai" | "function";
  sources?: Source[];
  name?: string;
  function_call?: { name: string };
};

export type Feedback = {
  feedback_id: string;
  score: number;
  comment?: string;
};

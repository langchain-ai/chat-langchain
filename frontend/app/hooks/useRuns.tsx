export function useRuns() {
  /**
   * Generates a public shared run ID for the given run ID.
   */
  const shareRun = async (runId: string): Promise<string | undefined> => {
    const res = await fetch("/api/runs/share", {
      method: "POST",
      body: JSON.stringify({ runId }),
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!res.ok) {
      return;
    }

    const { sharedRunURL } = await res.json();
    return sharedRunURL;
  };

  return {
    shareRun,
  };
}

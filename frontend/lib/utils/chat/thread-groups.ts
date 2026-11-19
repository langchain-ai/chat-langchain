export type GroupableThread = { updated_at?: string; created_at?: string }

export const THREAD_GROUP_TITLES = ["Today", "Yesterday", "Previous 7 Days", "Older"] as const

// Group by calendar day in the viewer's local time zone, not by rolling 24h
// windows: a chat from 9pm last night belongs under "Yesterday" at 9am today.
export function groupThreads<T extends GroupableThread>(threads: T[], now: Date = new Date()) {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfYesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime()
  const startOfLastWeek = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 7).getTime()
  const groups: { title: string; threads: T[] }[] = THREAD_GROUP_TITLES.map((title) => ({ title, threads: [] }))
  for (const thread of threads) {
    const ts = new Date(thread.updated_at || thread.created_at || 0).getTime()
    const group = ts >= startOfToday ? 0 : ts >= startOfYesterday ? 1 : ts >= startOfLastWeek ? 2 : 3
    groups[group].threads.push(thread)
  }
  return groups.filter((group) => group.threads.length > 0)
}

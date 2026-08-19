import assert from "node:assert/strict"
import test from "node:test"
import {
  claimSignInTracking,
  getSignInFingerprint,
  rememberSignInTracking,
} from "../lib/auth/sign-in-tracking.ts"

class MemoryStorage {
  private readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }
}

class SerialLocks {
  private readonly tails = new Map<string, Promise<void>>()

  async request(
    name: string,
    callback: () => boolean | Promise<boolean>
  ): Promise<boolean> {
    const previous = this.tails.get(name) ?? Promise.resolve()
    let release = () => {}
    const current = new Promise<void>((resolve) => {
      release = resolve
    })
    this.tails.set(name, previous.then(() => current))
    await previous

    try {
      return await callback()
    } finally {
      release()
    }
  }
}

const user = {
  id: "user-1",
  last_sign_in_at: "2026-08-18T12:00:00.000Z",
}

test("seeds an existing session without counting a later refocus", async () => {
  const storage = new MemoryStorage()

  await rememberSignInTracking("us", user, storage)
  assert.equal(await claimSignInTracking("us", user, storage), false)
})

test("deduplicates a genuine sign-in across refreshes without Web Locks", async () => {
  const storage = new MemoryStorage()

  assert.equal(await claimSignInTracking("us", user, storage), true)
  assert.equal(await claimSignInTracking("us", user, storage), false)
})

test("serializes concurrent claims when Web Locks are available", async () => {
  const storage = new MemoryStorage()
  const locks = new SerialLocks()

  const claims = await Promise.all([
    claimSignInTracking("us", user, storage, locks),
    claimSignInTracking("us", user, storage, locks),
  ])

  assert.deepEqual(claims.sort(), [false, true])
})

test("tracks later logins and rejects stale sessions", async () => {
  const storage = new MemoryStorage()
  const laterUser = {
    ...user,
    last_sign_in_at: "2026-08-18T13:00:00.000Z",
  }

  assert.equal(await claimSignInTracking("us", user, storage), true)
  assert.equal(await claimSignInTracking("us", laterUser, storage), true)
  assert.equal(await claimSignInTracking("us", user, storage), false)
  assert.equal(await claimSignInTracking("us", laterUser, storage), false)
})

test("keeps tracking independent by region and user", async () => {
  const storage = new MemoryStorage()

  assert.equal(await claimSignInTracking("us", user, storage), true)
  assert.equal(await claimSignInTracking("eu", user, storage), true)
  assert.equal(
    await claimSignInTracking("us", { ...user, id: "user-2" }, storage),
    true
  )
  assert.equal(await claimSignInTracking("us", user, storage), false)
})

test("allows one in-memory event when storage is unavailable", async () => {
  const storage = {
    getItem(): string | null {
      throw new Error("storage unavailable")
    },
    setItem(): void {
      throw new Error("storage unavailable")
    },
  }
  const originalWarn = console.warn
  let warning = ""
  console.warn = (message) => {
    warning = String(message)
  }

  try {
    assert.equal(await claimSignInTracking("us", user, storage), true)
  } finally {
    console.warn = originalWarn
  }

  assert.match(warning, /storage unavailable/)
  assert.equal(
    getSignInFingerprint("us", user),
    "us:user-1:2026-08-18T12:00:00.000Z"
  )
})

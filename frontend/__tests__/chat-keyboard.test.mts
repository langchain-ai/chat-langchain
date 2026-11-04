import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { runInNewContext } from "node:vm"
import ts from "typescript"

function readHandlers(file: string): ts.ArrowFunction[] {
  const source = ts.createSourceFile(
    file,
    readFileSync(new URL(`../components/chat/${file}`, import.meta.url), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  const handlers: ts.ArrowFunction[] = []
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "handleKeyDown" &&
      node.initializer &&
      ts.isCallExpression(node.initializer)
    ) {
      const callback = node.initializer.arguments[0]
      assert.ok(ts.isArrowFunction(callback))
      handlers.push(callback)
    }
    if (
      ts.isJsxAttribute(node) &&
      node.name.getText(source) === "onKeyDown" &&
      node.initializer &&
      ts.isJsxExpression(node.initializer) &&
      node.initializer.expression &&
      ts.isArrowFunction(node.initializer.expression)
    ) {
      handlers.push(node.initializer.expression)
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return handlers
}

const chatHandlers = readHandlers("chat-interface.tsx")
const messageHandlers = readHandlers("message-item.tsx")
assert.equal(chatHandlers.length, 1)
assert.equal(messageHandlers.length, 2)

for (const [name, handler] of [
  ["chat input", chatHandlers[0]],
  ["user message edit", messageHandlers[0]],
  ["feedback comment", messageHandlers[1]],
] as const) {
  test(`${name} leaves composition and Shift+Enter alone, then submits on Enter`, () => {
    let submissions = 0
    let cancellations = 0
    const submit = () => { submissions += 1 }
    const cancel = () => { cancellations += 1 }
    const code = ts.transpile(`(${handler.getText()})`, {
      target: ts.ScriptTarget.ES2017,
    })
    const onKeyDown = runInNewContext(code, {
      userId: "user-1",
      handleSend: submit,
      handleSaveEdit: submit,
      onSubmitComment: submit,
      handleCancelEdit: cancel,
      onCancelComment: cancel,
      feedbackComment: { "message-1": "Useful answer" },
      message: { id: "message-1", feedback: "positive" },
    })
    function press(overrides: Record<string, unknown> = {}) {
      let prevented = false
      onKeyDown({
        key: "Enter",
        shiftKey: false,
        keyCode: 13,
        nativeEvent: { isComposing: false },
        preventDefault() { prevented = true },
        ...overrides,
      })
      return prevented
    }

    assert.equal(press({ nativeEvent: { isComposing: true } }), false)
    assert.equal(press({ keyCode: 229 }), false)
    assert.equal(press({ key: "Process", keyCode: 229 }), false)
    assert.equal(press({ shiftKey: true }), false)
    assert.equal(press({ key: "a", keyCode: 65 }), false)
    assert.equal(submissions, 0)
    assert.equal(cancellations, 0)

    assert.equal(press(), true)
    assert.equal(submissions, 1)
    assert.equal(press({ key: "Escape", keyCode: 27 }), false)
    assert.equal(cancellations, name === "chat input" ? 0 : 1)
  })
}

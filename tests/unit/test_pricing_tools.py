"""Tests for readable pricing evidence returned by the pricing tool."""

import asyncio

import httpx

from src.tools import pricing_tools

PRICING_HTML = """
<html><head>
  <script>misleading pricing</script><style>.hidden { display: none; }</style>
</head><body>
  <p>Plans &amp; hosting&nbsp;options</p>

  <table>
    <tr><th>Feature</th><th>Developer</th><th>Plus</th><th>Enterprise</th></tr>
    <tr>
      <th>Hosting options</th>
      <td><div>Cloud</div></td>
      <td>\n Cloud \n</td>
      <td><div>Cloud,</div><div>Hybrid, or <strong>Self-Hosted</strong></div></td>
    </tr>
    <tr><th>Support</th><td></td><td>Standard</td><td>Custom &#38; priority</td></tr>
  </table>
  <p>Check &lt;license&gt; &#8212; don&#39;t guess &amp;lt;terms&amp;gt;.</p>
</body></html>
"""


def test_extract_text_preserves_pricing_columns():
    """Preserve plan columns, decoded entities, and normalized visible text."""
    text = pricing_tools._extract_text(PRICING_HTML)

    assert text.splitlines() == [
        "Plans & hosting options",
        "Feature | Developer | Plus | Enterprise",
        "Hosting options | Cloud | Cloud | Cloud, Hybrid, or Self-Hosted",
        "Support | | Standard | Custom & priority",
        "Check <license> — don't guess &lt;terms&gt;.",
    ]


def test_fetch_pricing_returns_plan_hosting_evidence(monkeypatch):
    """Return plan-aligned hosting evidence through the pricing tool."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=PRICING_HTML)
    )
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(pricing_tools.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(pricing_tools, "_cached_text", None)
    monkeypatch.setattr(pricing_tools, "_cached_at", 0.0)

    result = asyncio.run(pricing_tools.fetch_langchain_pricing.ainvoke({}))

    assert (
        "Feature | Developer | Plus | Enterprise\n"
        "Hosting options | Cloud | Cloud | Cloud, Hybrid, or Self-Hosted"
    ) in result

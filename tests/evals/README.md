# Chat LangChain regression set

`regression_set_v1.json` holds 16 cases taken from real prod traces, with the prod baseline verdict for each (build `866b859`, 2026-11-10: 8 pass, 1 pass-weak, 7 fail).

How to use it: load `cases` as a LangSmith dataset (input = `question`, reference = `expect`), run the agent on a candidate branch, and compare each case verdict to `baseline_2026_11_10.verdict`. A change should not turn a pass into a fail, and should fix the fails it claims to fix.

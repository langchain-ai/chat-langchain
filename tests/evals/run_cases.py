"""Run the regression questions against a Chat LangChain deployment via the public guest endpoint.
usage: python run_cases.py <deployment_url> <out.json>"""
import sys, json, time, re, urllib.request, concurrent.futures as cf
U=sys.argv[1].rstrip("/"); OUT=sys.argv[2]
Q={
"clc-01":"Tôi thấy có 2 repo github về open sandbox khác nhau. Nên dùng cái nào cho LangChain deep agents?",
"clc-02":"اين تقع DirectoryLoader",
"clc-04":"Wo kann ich den Artikel 'Attention is all you need' als pdf-file frei herunterladen?",
"clc-05":"google",
"clc-06":"how to use duck duckgo search tool",
"clc-08":"model.with_structured_output",
"clc-09":"checkpointer=True - does this use the parent's checkpointer in a subgraph?",
"clc-10":"Can I build a secure agentic workflow handling 30 million parallel requests with LangGraph?",
"clc-11":"do you confirm that this is new in LangSmith self-hosted v0.18? it used to work on v0.17",
"clc-12":"langsmith v0.18 does not seem to support OAuth HS512 token signing",
"clc-13":"how install mda (managed deep agents) with npm",
"clc-14":"link how to enable sandboxes in self host langsmith",
"clc-15":"how install @langchain/openrouter with npm",
"clc-16":"How do I add persistence to a LangGraph agent?",
"clc-17":"如何在 LangGraph 中配置 checkpointer?",
}
def post(path, body, tok=None, timeout=240):
    req=urllib.request.Request(U+path, data=json.dumps(body).encode(), headers={"content-type":"application/json", **({"Authorization":"Bearer "+tok} if tok else {})})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def status(u):
    try:
        r=urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"}); return urllib.request.urlopen(r, timeout=15).status
    except urllib.error.HTTPError as e: return e.code
    except Exception as e: return "ERR"
def run(item):
    cid,q=item; t0=time.time()
    try:
        tok=post("/identity/guest",{})["token"]
        r=post("/runs/wait",{"assistant_id":"docs-agent","input":{"messages":[{"role":"user","content":q}]}},tok)
        ai=[m for m in r.get("messages",[]) if m.get("type")=="ai" and m.get("content")][-1]["content"]
        if isinstance(ai,list): ai=" ".join(x.get("text","") for x in ai if isinstance(x,dict))
        urls=sorted(set(re.findall(r'\]\((https?://[^)\s]+)\)',ai)))
        return cid,{"q":q,"secs":round(time.time()-t0,1),"answer":ai,"links":{u:status(u) for u in urls}}
    except Exception as e:
        return cid,{"q":q,"error":str(e)[:200]}
with cf.ThreadPoolExecutor(5) as ex: res=dict(ex.map(run,Q.items()))
json.dump(res,open(OUT,"w"),indent=1,ensure_ascii=False)
for k,v in sorted(res.items()):
    print(k, v.get("secs"), v.get("error",""), {u[-50:]:s for u,s in v.get("links",{}).items() if s!=200})

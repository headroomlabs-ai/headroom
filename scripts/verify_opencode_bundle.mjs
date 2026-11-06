import assert from 'node:assert/strict';
import http from 'node:http';
import {pathToFileURL, fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(process.argv[2] ?? fileURLToPath(new URL('../plugins/opencode/', import.meta.url)))+path.sep;
const mod=await import(pathToFileURL(root+'dist/index.js'));
const standalone=await import(pathToFileURL(root+'dist-standalone/index.js'));
assert.equal(standalone.default.id,'headroom');
const seen=[];
const server=http.createServer(async(req,res)=>{let body='';for await(const c of req)body+=c;seen.push({url:req.url,headers:req.headers,body});res.setHeader('content-type','application/json');res.end(JSON.stringify(req.url.startsWith('/v1/retrieve/')?{content:'original fixture content'}:{choices:[{message:{content:'loopback response'}}]}));});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const proxyUrl=`http://127.0.0.1:${server.address().port}`;
let hooks;
try {
 hooks=await mod.default.server({directory:'isolated-smoke',project:{id:'smoke-project'}},{proxyUrl,sessionToken:'smoke-session'});
 const response=await fetch('https://api.openai.com/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'fixture',messages:[{role:'user',content:'local smoke'}]})});
 assert.equal((await response.json()).choices[0].message.content,'loopback response');
 assert.equal(seen[0].headers['x-headroom-base-url'],'https://api.openai.com');
 assert.equal(seen[0].headers['x-headroom-original-path'],'/v1/chat/completions');
 assert.equal(seen[0].headers['x-headroom-project'],'smoke-project');
 assert.equal(seen[0].headers['x-headroom-session-token'],'smoke-session');
 const retrieved=await hooks.tool.headroom_retrieve.execute({hash:'0123456789abcdef01234567'});
 assert.equal(JSON.parse(retrieved).content,'original fixture content');
 assert.equal(seen.length,2);
 console.log(JSON.stringify({node:process.version,artifacts:['dist/index.js','dist-standalone/index.js'],requests:seen.map(x=>({url:x.url,project:x.headers['x-headroom-project']})),result:'PASS: actual built plugin entry, transport interception, routing headers, JSON response and retrieve over real loopback HTTP'}));
} finally {await hooks?.dispose();server.closeAllConnections();await new Promise(r=>server.close(r));}
